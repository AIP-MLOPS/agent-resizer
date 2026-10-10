"""agent-resizer — modify a user agent's compute resources (CPU / RAM / MIG GPU).

Every MLOPS user gets a dedicated clearml-agent Deployment
`<username>-agent-clearml-agent` in namespace `user-<username>`, created at
provisioning time by agent-creation + user-management-v2. The critical resource
is a MIG GPU slice (e.g. `nvidia.com/mig-1g.10gb: 1`) on the agent container
(`k8s-glue`), alongside cpu/memory requests + limits.

This service patches that container's `resources` block *in place* via the K8s
API — a surgical change that triggers a rolling restart of the agent pod with
the new limits. It does NOT touch the agent's clearml.conf / credentials /
queue, and it NEVER creates or deletes the agent.

Endpoints:
  GET  /health
  GET  /agents/{username}/resources          -> current resources
  PATCH/agents/{username}/resources          -> change any of cpu/ram/gpu

Namespace/name conventions match agent-creation + user-management:
  namespace = user-<normalized_username>
  deployment = <normalized_username>-agent-clearml-agent
  container  = k8s-glue
All three are overridable via env for forward-compat.
"""
import logging
import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent-resizer")

# ── Conventions (env-overridable) ─────────────────────────────────────────────
NS_PREFIX = os.getenv("AGENT_NS_PREFIX", "user-")
DEPLOY_SUFFIX = os.getenv("AGENT_DEPLOY_SUFFIX", "-agent-clearml-agent")
CONTAINER_NAME = os.getenv("AGENT_CONTAINER_NAME", "k8s-glue")
# Optional bearer auth: if set, every non-health request must present
# `Authorization: Bearer <RESIZER_AUTH_TOKEN>`. Empty => no auth (rely on the
# in-cluster ingress boundary). Wire it from a Secret in the chart for prod.
AUTH_TOKEN = os.getenv("RESIZER_AUTH_TOKEN", "").strip()
# The user-services ingress forwards the FULL path (no StripPrefix), so we mount
# the user-facing routes under this prefix; requests arrive as
# /agent-resources/agents/<u>/resources. /health stays at the root for kubelet
# probes (which hit the container port directly, without the prefix).
API_PREFIX = os.getenv("API_PREFIX", "").rstrip("/")

app = FastAPI(title="agent-resizer", docs_url="/docs")
router = APIRouter(prefix=API_PREFIX)
_bearer = HTTPBearer(auto_error=False)


@app.on_event("startup")
def _load_kube():
    try:
        config.load_incluster_config()
        logger.info("loaded in-cluster kube config")
    except Exception:
        config.load_kube_config()  # local dev convenience
        logger.info("loaded local kube config (~/.kube/config)")
    if not AUTH_TOKEN:
        logger.warning(
            "RESIZER_AUTH_TOKEN is unset — the resize API is UNAUTHENTICATED. "
            "Set it (from a Secret) before exposing this service publicly."
        )


def _require_auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if not AUTH_TOKEN:
        return
    if creds is None or creds.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def normalize_username(username: str) -> str:
    """Same rule agent-creation / user-management apply: lowercase, keep [0-9a-z]."""
    return re.sub(r"[^0-9a-z]", "", (username or "").lower())


def _coords(username: str) -> tuple[str, str]:
    u = normalize_username(username)
    if not u:
        raise HTTPException(status_code=400, detail="invalid username")
    namespace = f"{NS_PREFIX}{u}"
    deployment = f"{u}{DEPLOY_SUFFIX}"
    return namespace, deployment


class ResourcePatch(BaseModel):
    """All fields optional — only what you set is changed.

    GPU semantics:
      - gpu_profile: the MIG profile suffix, i.e. the part after `nvidia.com/`
        (e.g. "mig-1g.10gb", or "gpu" for a whole GPU). Swaps the resource key.
      - gpu_count:   how many of that resource (0 => REMOVE the GPU entirely).
    """
    cpu_request: Optional[str] = Field(None, examples=["2", "500m"])
    ram_request: Optional[str] = Field(None, examples=["4Gi"])
    cpu_limit: Optional[str] = Field(None, examples=["4"])
    ram_limit: Optional[str] = Field(None, examples=["8Gi"])
    gpu_profile: Optional[str] = Field(None, examples=["mig-1g.10gb"])
    gpu_count: Optional[int] = Field(None, ge=0, examples=[1])

    @model_validator(mode="after")
    def _at_least_one(self):
        if all(
            getattr(self, f) is None
            for f in ("cpu_request", "ram_request", "cpu_limit", "ram_limit",
                      "gpu_profile", "gpu_count")
        ):
            raise ValueError("provide at least one field to change")
        return self


def _apps() -> client.AppsV1Api:
    return client.AppsV1Api()


def _get_agent_container(namespace: str, deployment: str):
    apps = _apps()
    try:
        dep = apps.read_namespaced_deployment(deployment, namespace)
    except ApiException as e:
        if e.status == 404:
            raise HTTPException(
                status_code=404,
                detail=f"agent deployment '{deployment}' not found in namespace '{namespace}'",
            )
        raise HTTPException(status_code=502, detail=f"k8s read failed: {e.status} {e.reason}")
    containers = dep.spec.template.spec.containers or []
    for c in containers:
        if c.name == CONTAINER_NAME:
            return dep, c
    raise HTTPException(
        status_code=404,
        detail=f"container '{CONTAINER_NAME}' not found on {deployment}; "
               f"have {[c.name for c in containers]}",
    )


def _resources_view(container) -> dict:
    res = container.resources
    requests = dict((res.requests or {})) if res else {}
    limits = dict((res.limits or {})) if res else {}
    gpu = {k: v for k, v in limits.items() if k.startswith("nvidia.com/")}
    return {
        "container": container.name,
        "requests": requests,
        "limits": limits,
        "gpu": gpu or None,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/agents/{username}/resources", dependencies=[Depends(_require_auth)])
def get_resources(username: str):
    namespace, deployment = _coords(username)
    _, container = _get_agent_container(namespace, deployment)
    return {
        "username": normalize_username(username),
        "namespace": namespace,
        "deployment": deployment,
        **_resources_view(container),
    }


@router.patch("/agents/{username}/resources", dependencies=[Depends(_require_auth)])
def patch_resources(username: str, patch: ResourcePatch):
    namespace, deployment = _coords(username)
    _, container = _get_agent_container(namespace, deployment)

    cur_res = container.resources
    cur_limits = dict((cur_res.limits or {})) if cur_res else {}
    cur_nvidia = {k: v for k, v in cur_limits.items() if k.startswith("nvidia.com/")}

    req_patch: dict = {}
    lim_patch: dict = {}
    if patch.cpu_request is not None:
        req_patch["cpu"] = patch.cpu_request
    if patch.ram_request is not None:
        req_patch["memory"] = patch.ram_request
    if patch.cpu_limit is not None:
        lim_patch["cpu"] = patch.cpu_limit
    if patch.ram_limit is not None:
        lim_patch["memory"] = patch.ram_limit

    # ── GPU / MIG ──
    if patch.gpu_profile is not None or patch.gpu_count is not None:
        # Resolve the target profile + count from the request, falling back to
        # whatever the agent currently has.
        cur_key = next(iter(cur_nvidia), None)                 # e.g. nvidia.com/mig-1g.5gb
        cur_profile = cur_key.split("nvidia.com/", 1)[1] if cur_key else None
        cur_count = int(cur_nvidia[cur_key]) if cur_key else 0

        profile = patch.gpu_profile if patch.gpu_profile not in (None, "") else cur_profile
        count = patch.gpu_count if patch.gpu_count is not None else (cur_count or 1)

        # Strategic-merge deletes map keys set to null — null out every existing
        # nvidia.com/* key so a profile SWAP doesn't leave the old resource.
        for k in cur_nvidia:
            lim_patch[k] = None
        if count > 0 and profile:
            lim_patch[f"nvidia.com/{profile}"] = count
        elif count > 0 and not profile:
            raise HTTPException(
                status_code=400,
                detail="gpu_count > 0 but no gpu_profile and the agent has no existing GPU to infer from",
            )
        # count == 0 => GPU removed (old keys nulled above, nothing re-added)

    resources_patch: dict = {}
    if req_patch:
        resources_patch["requests"] = req_patch
    if lim_patch:
        resources_patch["limits"] = lim_patch

    body = {
        "spec": {"template": {"spec": {"containers": [
            {"name": CONTAINER_NAME, "resources": resources_patch},
        ]}}}
    }
    logger.info("patching %s/%s container=%s resources=%s", namespace, deployment, CONTAINER_NAME, resources_patch)
    try:
        # Strategic-merge patch: merges the container by name (patchMergeKey),
        # updates provided resource keys, and deletes keys set to null.
        _apps().patch_namespaced_deployment(deployment, namespace, body)
    except ApiException as e:
        # 422 = bad quantity / invalid resource; surface k8s' message verbatim.
        raise HTTPException(status_code=e.status or 502, detail=f"k8s patch failed: {e.reason} — {e.body}")

    # Re-read so the caller sees the effective new spec.
    _, new_container = _get_agent_container(namespace, deployment)
    return {
        "username": normalize_username(username),
        "namespace": namespace,
        "deployment": deployment,
        "status": "patched — agent pod will roll to the new resources",
        **_resources_view(new_container),
    }


app.include_router(router)

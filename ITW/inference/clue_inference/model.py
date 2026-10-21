"""SAM 3D Body, SAM 3D Body (CLUE) and CLUE: the MHR mesh of every person crop, and for the CLUE models the uncertainty
of each of its vertices.

All three have SAM 3D Body's architecture and share its DINOv3 backbone:
* SAM 3D Body: Meta's released model (facebook/sam-3d-body-dinov3: model.ckpt and the MHR rig assets/mhr_model.pt).
* SAM 3D Body (CLUE): the released model, unchanged, with CLUE's uncertainty head trained on top of it
  (sam3dbody_clue.safetensors); its mesh is SAM 3D Body's mesh.
* CLUE: the released backbone and MHR head constants with CLUE's decoder and heads and its own uncertainty head
  (clue.safetensors).
An uncertainty head predicts variances of the MHR parameters from the decoder's pose token and the predicted parameters;
CLUE propagates them through the rig in closed form to a variance per vertex (plugin/sam3dbody_clue/head.py, shared with
the standalone SAM 3D Body (CLUE) plugin). The released decoder runs once for SAM 3D Body and SAM 3D Body (CLUE); the
backbone runs once per batch (``SharedBackbone``). All predict hands and face, as in our evaluations. The model files are
read from the assets folder (assets.py).

The heads' mesh faces stay on the CPU (the vendored head copies them to numpy after every decoder layer); the rig's
TorchScript fuser is off and, on Blackwell GPUs, cuDNN attention (``configure_gpu``).
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT, ROOT / "plugin"):              # the vendored sam_3d_body package and the plugin's sam3dbody_clue
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from clue_inference.assets import ASSETS, model_file  # noqa: E402

MODELS = ("sam3dbody", "sam3dbody_clue", "clue")
TITLES = {"sam3dbody": "SAM 3D Body", "sam3dbody_clue": "SAM 3D Body (CLUE)", "clue": "CLUE"}
DINOV3_COMMIT = "31703e4cbf1ccb7c4a72daa1350405f86754b6d1"    # facebookresearch/dinov3, the version CLUE was evaluated with
BODY_SHAPE = slice(0, 20)                           # MHR identity parameters: [0:20] body shape, [20:40] head, [40:45] hands
BATCH_TENSOR_KEYS = ("img", "img_size", "ori_img_size", "bbox_center", "bbox_scale", "bbox", "affine_trans", "mask", "mask_score")


@dataclass
class Person:
    track_id: int
    box: np.ndarray                     # (4,) xyxy in pixels: the box the models were given
    sam3d: np.ndarray | None            # (V, 3) SAM 3D Body's mesh (also SAM 3D Body (CLUE)'s) in metres, camera frame
                                        # with y up and visible depth at -z
    sam3d_sigma_cm: np.ndarray | None   # (V,) SAM 3D Body (CLUE): sigma_total = sqrt(var_x + var_y + var_z) in cm of that
                                        # mesh's vertices, root rotation / translation excluded
    clue: np.ndarray | None             # (V, 3) CLUE's mesh, same frame
    clue_sigma_cm: np.ndarray | None    # (V,) CLUE: sigma_total of its mesh's vertices, as above
    clue_shape_sigma_cm: np.ndarray | None = None   # (V,) CLUE: the part of that sigma due to body shape alone (MHR identity
                                                    # parameters 0-19); only with predict(..., shape=True)

    @property
    def vertices(self) -> np.ndarray:
        return self.clue if self.clue is not None else self.sam3d


def cliff_intrinsics(width: int, height: int, rect: np.ndarray | None = None) -> np.ndarray:
    """Camera approximation of the CLUE evaluations: focal length = picture diagonal, principal point = picture centre,
    taken on the active picture ``rect`` (x0, y0, x1, y1) of letterboxed frames, the whole frame otherwise."""
    x0, y0, x1, y1 = (0.0, 0.0, float(width), float(height)) if rect is None else (float(v) for v in rect)
    focal = float(np.hypot(x1 - x0, y1 - y0))
    return np.asarray([[focal, 0.0, (x0 + x1) / 2.0], [0.0, focal, (y0 + y1) / 2.0], [0.0, 0.0, 1.0]], np.float32)


def configure_gpu(device: torch.device) -> None:
    """Process-wide switches, applied before the first forward.
    * TorchScript fuser off: it compiles fused kernels of the MHR rig for every new batch shape, which costs more than it
      saves here, cold and warm (H100: one image 0.85 s instead of 1.31 s on the first call, 68 people 0.83 s instead of
      1.29 s warm; identical output), and it fails where NVRTC does not know the GPU (B300, sm_103).
    * Blackwell (sm_100+): cuDNN attention off; torch 2.11+cu128 builds no execution plan for the DINOv3 attention there."""
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 10:
        torch.backends.cuda.enable_cudnn_sdp(False)


def dinov3_code() -> Path:
    """DINOv3's model code (DINOv3 License) at a fixed commit, downloaded from GitHub once into torch's hub cache. The
    backbone module is imported from it directly (sam_3d_body/models/backbones/dinov3.py): hubconf.py on DINOv3's main
    branch also imports its segmentation and depth heads, which need packages this code does not install."""
    import io
    import shutil
    import urllib.request
    import zipfile

    target = Path(torch.hub.get_dir()) / f"facebookresearch_dinov3_{DINOV3_COMMIT[:12]}"
    if not (target / "dinov3/hub/backbones.py").exists():
        url = f"https://github.com/facebookresearch/dinov3/archive/{DINOV3_COMMIT}.zip"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        archive.extractall(staging)
        shutil.rmtree(target, ignore_errors=True)
        (staging / f"dinov3-{DINOV3_COMMIT}").rename(target)
        shutil.rmtree(staging, ignore_errors=True)
    return target


class SharedBackbone(torch.nn.Module):
    """The DINOv3 backbone of both decoders: the first decoder run on a batch computes the image embeddings, the second
    (``reuse``) takes them over."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.reuse, self.embeddings = False, None

    def forward(self, x: torch.Tensor, extra_embed: torch.Tensor | None = None):
        if not self.reuse:
            self.embeddings = self.backbone(x, extra_embed=extra_embed)
        return self.embeddings


def decoder_copy(body: torch.nn.Module) -> torch.nn.Module:
    """A copy of ``body`` that shares its backbone and MHR rig (the rig holds constants only)."""
    backbone, rig = body.backbone, body.head_pose.mhr
    body.backbone, body.head_pose.mhr = None, None
    try:
        twin = copy.deepcopy(body)
    finally:
        body.backbone, body.head_pose.mhr = backbone, rig
    twin.backbone, twin.head_pose.mhr = backbone, rig
    return twin


class Models:
    """SAM 3D Body, SAM 3D Body (CLUE) and/or CLUE (``models``, a subset of MODELS) on ``device``, from the model files in
    ``assets``."""

    def __init__(self, models: tuple[str, ...] = MODELS, device: torch.device | str = "cuda", *, assets: Path = ASSETS) -> None:
        from safetensors.torch import load_file

        from sam3dbody_clue.head import VertexUncertainty
        from sam_3d_body import SAM3DBodyEstimator
        from sam_3d_body.build_models import load_sam_3d_body

        if not models or set(models) - set(MODELS):
            raise ValueError(f"models must be a non-empty subset of {MODELS}, got {models}")
        self.models = tuple(m for m in MODELS if m in models)
        self.device = torch.device(device)
        configure_gpu(self.device)
        if not os.environ.get("SAM3D_DINOV3_REPO"):                       # checked first by the vendored backbone loader
            os.environ["SAM3D_DINOV3_REPO"] = str(dinov3_code())
        body, cfg = load_sam_3d_body(checkpoint_path=str(model_file("checkpoint", assets)), config_path=str(ROOT / "model_config.yaml"),
                                     mhr_path=str(model_file("rig", assets)), checkpoint_load_mode="full", device=self.device)
        body.head_pose.predict_hands_face = True
        released = {"sam3dbody", "sam3dbody_clue"} & set(self.models)
        self.sam3d = body if released else None                           # the released decoder
        self.clue, self.heads = None, {}
        if "clue" in self.models:
            clue = decoder_copy(body) if released else body
            state = load_file(str(model_file("clue", assets)), device="cpu")
            clue_state = {k[len("body_model."):]: v for k, v in state.items() if k.startswith("body_model.")}
            unknown = [k for k in clue_state if k not in clue.state_dict()]
            if unknown:
                raise RuntimeError(f"clue.safetensors does not match the SAM 3D Body architecture: {unknown[:3]}")
            clue.load_state_dict(clue_state, strict=False)
            self.heads["clue"] = self._head(state, cfg, body)
            self.clue = clue
        if "sam3dbody_clue" in self.models:
            self.heads["sam3dbody_clue"] = self._head(load_file(str(model_file("sam3dbody_clue", assets)), device="cpu"), cfg, body)
        if self.heads:                                                    # one rig, shared by all models
            self.uncertainty = VertexUncertainty(body.head_pose.mhr, self.device)
        self.backbone = SharedBackbone(body.backbone)
        for _, net in self.networks():
            net.backbone = self.backbone
            net.eval()
            net.head_pose.faces = torch.nn.Parameter(net.head_pose.faces.detach().cpu(), requires_grad=False)
        self.transform = SAM3DBodyEstimator(body, cfg).transform
        self.faces = np.asarray(body.head_pose.mhr.character_torch.mesh.faces.detach().cpu(), dtype=np.int32)

    def _head(self, state: dict, cfg, body: torch.nn.Module) -> torch.nn.Module:
        from sam3dbody_clue.head import VarianceHead

        head = VarianceHead(token_dim=int(cfg.MODEL.DECODER.DIM), num_scale_comps=int(body.head_pose.num_scale_comps))
        head.load_state_dict({k[len("variance_head."):]: v for k, v in state.items() if k.startswith("variance_head.")})
        return head.to(self.device).eval()

    def networks(self, models: tuple[str, ...] | None = None) -> list[tuple[str, torch.nn.Module]]:
        """The decoders that ``models`` (default: all loaded) need: the released one (SAM 3D Body, SAM 3D Body (CLUE))
        and CLUE's."""
        models = self.models if models is None else models
        wanted = {"sam3d": bool({"sam3dbody", "sam3dbody_clue"} & set(models)), "clue": "clue" in models}
        return [(name, net) for name, net in (("sam3d", self.sam3d), ("clue", self.clue)) if net is not None and wanted[name]]

    @torch.no_grad()
    def _forward(self, batch: dict, models: tuple[str, ...] | None = None, shape: bool = False) -> dict:
        """Per decoder: vertices and camera translation; per uncertainty head: the per-vertex variance (cm^2); with
        ``shape``, also CLUE's per-vertex variance from the body-shape parameters alone."""
        models = self.models if models is None else models
        from sam_3d_body.utils import recursive_to

        batch = recursive_to(batch, self.device)
        out = {}
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                for k, (name, net) in enumerate(self.networks(models)):
                    self.backbone.reuse = k > 0
                    net._initialize_batch(batch)
                    output = net.run_inference(None, batch, inference_type="body")
                    mhr = output["mhr"]
                    out[name] = (mhr["pred_vertices"], mhr["pred_cam_t"])
                    head = "clue" if name == "clue" else "sam3dbody_clue"
                    if head in self.heads and head in models:
                        variances = self.heads[head](output["pose_token"], mhr)
                        out[f"{name}_variance"] = self.uncertainty(mhr, variances)
                        if shape and name == "clue":
                            body_shape = torch.zeros_like(variances["shape"])
                            body_shape[:, BODY_SHAPE] = variances["shape"][:, BODY_SHAPE]
                            out["clue_shape_variance"] = self.uncertainty(
                                mhr, {"shape": body_shape, "body_pose": torch.zeros_like(variances["body_pose"])})
        finally:
            self.backbone.reuse, self.backbone.embeddings = False, None
        return out

    @torch.no_grad()
    def predict(self, frames_rgb: list[np.ndarray], people: list[list[tuple[int, np.ndarray]]], cam_ints: list[np.ndarray],
                max_batch: int = 64, workers: int = 8, prefetch: int = 16, models: tuple[str, ...] | None = None,
                shape: bool = False) -> list[list[Person]]:
        """``people[i]``: the (track id, xyxy box) pairs of frame i; ``cam_ints[i]``: its (3, 3) intrinsics; ``models``: a
        subset of the loaded models to run (default: all); ``shape``: also CLUE's uncertainty from body shape alone
        (``Person.clue_shape_sigma_cm``). The person crops are prepared on ``workers`` CPU threads up to ``prefetch`` frames
        ahead, overlapping the GPU forwards."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        from sam_3d_body.data.utils.prepare_batch import prepare_batch

        results: list[list[Person]] = [[] for _ in frames_rgb]
        pending, owners = [], []

        def prepare(i: int) -> tuple[int, dict]:
            persons = people[i]
            batch = prepare_batch(frames_rgb[i], self.transform, np.stack([box for _, box in persons]).astype(np.float32))
            part = {k: batch[k].squeeze(0).unsqueeze(1) for k in BATCH_TENSOR_KEYS}           # (P, 1, ...)
            part["cam_int"] = torch.from_numpy(np.asarray(cam_ints[i], np.float32))[None].expand(len(persons), -1, -1)
            return i, part

        def flush() -> None:
            if not owners:
                return
            batch = {k: torch.cat([p[k] for p in pending]) for k in BATCH_TENSOR_KEYS}
            batch["cam_int"] = torch.cat([p["cam_int"] for p in pending]).float()
            batch["person_valid"] = torch.ones((len(owners), 1))
            out = self._forward(batch, models, shape)
            arrays = {}
            for name, _ in self.networks(models):
                vertices, cam_t = out[name]
                vertices = vertices.float() + cam_t.float()[:, None, :]
                vertices[..., [1, 2]] *= -1                                       # y up, visible depth at negative z
                arrays[name] = vertices.cpu().numpy()
                if f"{name}_variance" in out:
                    arrays[f"{name}_sigma"] = out[f"{name}_variance"].float().clamp_min(0).sum(-1).sqrt().cpu().numpy()
            if "clue_shape_variance" in out:
                arrays["clue_shape_sigma"] = out["clue_shape_variance"].float().clamp_min(0).sum(-1).sqrt().cpu().numpy()
            for k, (i, track_id, box) in enumerate(owners):
                results[i].append(Person(track_id, np.asarray(box, np.float32),
                                         *(arrays[key][k] if key in arrays else None for key in ("sam3d", "sam3d_sigma", "clue", "clue_sigma")),
                                         clue_shape_sigma_cm=arrays["clue_shape_sigma"][k] if "clue_shape_sigma" in arrays else None))
            pending.clear(); owners.clear()

        todo = iter([i for i, persons in enumerate(people) if persons])
        with ThreadPoolExecutor(max(1, workers)) as pool:
            futures = deque(pool.submit(prepare, i) for _, i in zip(range(prefetch), todo))
            while futures:
                i, part = futures.popleft().result()
                following = next(todo, None)
                if following is not None:
                    futures.append(pool.submit(prepare, following))
                if owners and len(owners) + len(people[i]) > max_batch:
                    flush()
                pending.append(part)
                owners += [(i, track_id, box) for track_id, box in people[i]]
            flush()
        return results

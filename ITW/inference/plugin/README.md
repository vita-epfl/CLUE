# SAM 3D Body (CLUE): per-vertex uncertainty for SAM 3D Body

A plugin that adds CLUE's per-vertex uncertainty to [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) as you
already run it. Its head was trained on top of Meta's released SAM 3D Body, which stayed frozen. The plugin reads the
decoder's pose token and parameter predictions through a hook, and CLUE propagates the head's variances through the MHR
rig in closed form. SAM 3D Body's own outputs (meshes, keypoints, cameras) do not change.

Tested with Meta's [sam-3d-body](https://github.com/facebookresearch/sam-3d-body) at commit
[`b5c765a`](https://github.com/facebookresearch/sam-3d-body/commit/b5c765a0d89d789985e186d396315e7590887b94)
(2026-02-19) and the `facebook/sam-3d-body-dinov3` checkpoint.

## Install

This folder is self-contained: install it into the Python environment where SAM 3D Body runs,

```bash
pip install path/to/plugin
```

or copy `sam3dbody_clue/` next to your code. It needs `torch`, `numpy`, `safetensors` and `huggingface_hub`, which SAM 3D
Body already uses.

The head's weights (`sam3dbody_clue.safetensors`, 6 MB, SAM License) are downloaded from
[meghshukla/clue](https://huggingface.co/meghshukla/clue) on first use. The repository is gated with automatic approval:
accept its terms once and log in with `hf auth login`. You can also pass a downloaded file as `weights=`.

## Use with Meta's code

Attach the plugin once to your estimator; calls with the body decoder then return the uncertainty with SAM 3D Body's
usual outputs:

```python
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam3dbody_clue import SAM3DBodyCLUE

model, cfg = load_sam_3d_body(checkpoint_path, device="cuda", mhr_path=mhr_path)
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)      # as usual, with or without a human detector
clue = SAM3DBodyCLUE(estimator)                                              # once

outputs = estimator.process_one_image(image, inference_type="body")          # as usual, with the body decoder
```

Each person in `outputs` also has `outputs[i]["vertex_sigma_cm"]`: for each of the 18,439 vertices of
`outputs[i]["pred_vertices"]`, the standard deviation of its position in cm (root rotation and translation excluded).
Everything else is SAM 3D Body's own output: `process_one_image` keeps Meta's arguments, defaults and results. Person
boxes (`bboxes=`) are optional, as in Meta's code, which otherwise uses its human detector, or the whole image without
one.

`clue.sigma_cm()` returns the uncertainty of the last call as one `(people, 18439)` array, and `clue.variance_cm2()` the
per-axis variances `(people, 18439, 3)`. If you call the SAM 3D Body model yourself instead of the estimator, attach the
plugin to the model (`SAM3DBodyCLUE(model)`) and read the uncertainty of its last forward with these two. `clue.remove()`
detaches the plugin.

## Notes

- **Body decoder.** The uncertainty is added to calls with `inference_type="body"`: SAM 3D Body's body decoder, which
  predicts the whole body, hands and face included. The head was trained on that decoder, so the uncertainty describes
  exactly the returned mesh. Meta's default, `"full"`, adds a second decoder that re-estimates the hands from hand crops
  and replaces the hands and wrists of the final mesh, which the head has not seen; such calls (and `"hand"`) return
  SAM 3D Body's outputs unchanged, without uncertainty, with a one-time warning.
- **Hand-detection tokens.** Meta's config switches them on (`DO_HAND_DETECT_TOKENS: true`). CLUE's evaluations and
  inference code run SAM 3D Body with them off. With them off, the plugin on Meta's code gives exactly the meshes and
  uncertainty of CLUE's own inference code. With them on, as by default, SAM 3D Body's meshes change slightly (3 mm per
  vertex on average in our check), while the head's uncertainty changes by about 1% at most.
- **MHR rig.** The plugin uses SAM 3D Body's TorchScript rig (`assets/mhr_model.pt`). When SAM 3D Body runs the MHR Python
  package instead, the plugin loads the TorchScript rig from the same path.

## Contents and licence

| path | content |
|---|---|
| `sam3dbody_clue/plugin.py` | `SAM3DBodyCLUE`: the hook, the head and the uncertainty of SAM 3D Body's last forward |
| `sam3dbody_clue/head.py` | the variance head and the per-vertex propagation |
| `sam3dbody_clue/clue_mhr/` | CLUE for MHR: the closed-form propagation through the MHR rig |

The plugin's code is released under the BSD-3-Clause-Clear License (`LICENSE`), which grants no patent rights; the head's
weights under the SAM License. If you use it, please cite the paper:

```bibtex
@misc{shukla2026modelinguncertaintypropagation3d,
      title={Modeling Uncertainty Propagation in 3D Morphable Models}, 
      author={Megh Shukla and Edo Collins and Prashanth Chandran and Karel Petránek and Sergio Orts-Escolano and Mathieu Salzmann and Alexandre Alahi},
      year={2026},
      eprint={2610.24646},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.24646}, 
}
```

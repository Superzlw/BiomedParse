"""BiomedParse v1 单张图推理：任意 prompt，输出每个 prompt 的 mask 和叠加图。

在仓库根目录、biomedparse 环境里运行：
  python infer_prompt.py <图> <输出前缀> <TILE> "<prompt1>" ["<prompt2>" ...]
TILE=0 表示整张图直接推理（官方用法，1024 的 tile 就用 0）；TILE>0 表示按 TILE×TILE 切块推理再拼回。
多个 prompt 一次推理一起送进去（和官方评测一样），阈值 0.5。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats

if len(sys.argv) < 5:
    sys.exit(__doc__)
img_path, out_prefix, TILE, PROMPTS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
SUPPORTED = {"neoplastic cells", "inflammatory cells", "connective tissue cells", "epithelial cells"}  # 病理训练过、能算 p-value 的

opt = init_distributed(load_opt_from_config_files(["configs/biomedparse_inference.yaml"]))
model = BaseModel(opt, build_model(opt)).from_pretrained("hf_hub:microsoft/BiomedParse").eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

image = np.array(Image.open(img_path).convert("RGB"))
H, W = image.shape[:2]
step = TILE if TILE > 0 else max(H, W)
print(f"image {W}x{H}, tile {TILE if TILE > 0 else '整张图'}")
prob = np.zeros((len(PROMPTS), H, W), np.float32)
for y in range(0, H, step):
    for x in range(0, W, step):
        tile = Image.fromarray(image[y:y + step, x:x + step])
        prob[:, y:y + step, x:x + step] = interactive_infer_image(model, tile, PROMPTS)

os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
for i, p in enumerate(PROMPTS):
    m = prob[i] > 0.5
    tag = p.replace(" ", "+")
    Image.fromarray(m.astype(np.uint8) * 255).save(f"{out_prefix}_{tag}_mask.png")
    ovl = image.copy()
    ovl[m] = (0.5 * ovl[m] + [127, 0, 0]).astype(np.uint8)       # 红色半透明叠加
    Image.fromarray(ovl).save(f"{out_prefix}_{tag}_overlay.png")
    pv = f"{check_mask_stats(image, prob[i] * 255, 'Pathology', p):.4f}" if p in SUPPORTED else "n/a（未训练过的目标）"
    print(f"{p}: 前景占比 {m.mean():.3f}, p-value {pv}  →  {out_prefix}_{tag}_overlay.png")

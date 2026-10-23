from PIL import Image
import torch
import os
import shutil
from tqdm import tqdm
import random
import json
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
import numpy as np

# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
#pretrained_pth = 'pretrained/biomed_parse.pt'
pretrained_pth = "/home/zhilong/Github_Base/BiomedParse/pretrained/biomedparse_v1.pt"   # 'hf_hub:microsoft/BiomedParse'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)


tile_dir = ("/media/zhilong/bf844588-a56b-421b-b760-4505fcf10d28/VLM/Dataset/TCGA_LUAD/07_Tiles/"
            "TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01/SingleCell")
json_file = ("/media/zhilong/bf844588-a56b-421b-b760-4505fcf10d28/VLM/Dataset/TCGA_LUAD/08_Json_Output/ANNOTATION_L4_s/"
             "TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01_filtered.json")
save_dir = "/home/zhilong/Github_Base/BiomedParse/test_res/res_allcell/"

save_tumor_dir = "/home/zhilong/Github_Base/BiomedParse/test_res/res_allcell_tumor/"


img_dir = "/home/zhilong/Github_Base/BiomedParse/test_res/img/"

with open(json_file, "r") as f:
    json_info = json.load(f)

file_lst = list(json_info.keys())

# img_lst = [img_n[:-3]+'jpg' for img_n in file_lst]
prompts = ['neutrophils cells', 'tumor cells', 'lymphocytes cells', 'plasma cells', 'eosinophils cells',
           'connective cells', 'mitotic cells', 'benign cells', 'macrophage cells']
# sample_lst = random.sample(img_lst, 30)
sample_lst = os.listdir(img_dir)
for cur_img_n in tqdm(sample_lst):
    cur_img_path = os.path.join(tile_dir, cur_img_n)
    image = Image.open(cur_img_path)
    image = image.convert('RGB')
    all_pred_mask = interactive_infer_image(model, image, prompts)

    tumor_msk = all_pred_mask[1]
    binary_mask = (tumor_msk > 0.5).astype(np.uint8)

    Image.fromarray(binary_mask, mode="L").save(os.path.join(save_tumor_dir, cur_img_n[:-4]+'.png'))

    if image is not None:
        continue




    mask_thresh = all_pred_mask > 0.5
    final_mask = np.zeros((224, 224), dtype=np.uint8)
    argmax_labels = np.argmax(all_pred_mask, axis=0) + 1
    final_mask[np.any(mask_thresh, axis=0)] = argmax_labels[np.any(mask_thresh, axis=0)]

    # Image.fromarray(final_mask).save(os.path.join(save_dir, cur_img_n[:-4]+'.png'))














    # shutil.copy(cur_img_path, os.path.join(new_img_dir, cur_img_n))


# # RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
# image = Image.open('examples/Part_1_516_pathology_breast.png', formats=['png'])
# image = image.convert('RGB')
# # text prompts querying objects in the image. Multiple ones can be provided.
# prompts = ['neoplastic cells']
# # ['neoplastic cells', 'inflammatory cells']
#
# # load ground truth mask
# gt_masks = []
# for prompt in prompts:
#     gt_mask = Image.open(f"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png", formats=['png'])
#     gt_mask = 1*(np.array(gt_mask.convert('RGB'))[:,:,0] > 0)
#     gt_masks.append(gt_mask)
#
# pred_mask = interactive_infer_image(model, image, prompts)
#
# pred_mask = pred_mask.squeeze(0)
# binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
#
# img = Image.fromarray(binary_mask, mode="L")
# img.save("/home/zhilong/Github_Base/BiomedParse/test_res/example_color.png")
# # Image.fromarray(pred_mask).save("/home/zhilong/Github_Base/BiomedParse/test_res/example.png")
# # prediction with ground truth mask
# for i, pred in enumerate(pred_mask):
#     gt = gt_masks[i]
#     dice = (1*(pred>0.5) & gt).sum() * 2.0 / (1*(pred>0.5).sum() + gt.sum())
#     print(f'Dice score for {prompts[i]}: {dice:.4f}')
#     check_mask_stats(image, pred_mask[i]*255, 'X-Ray-Chest', text_prompt[i])
#     print(f'p-value for {prompts[i]}: {p_value:.4f}')
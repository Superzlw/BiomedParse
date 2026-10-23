import os
import cv2
from PIL import Image
import numpy as np


Tumor_Cell_Color = [
    [0, 0, 0],
    [255, 0, 0],  # tumor
]

color_cell = [
    # [0, 0, 0],
    [0, 255, 0],  # neu
    [255, 0, 0],  # tumor
    [0, 0, 255],  # lym
    [0, 128, 0],  # pla
    [0, 255, 255],  # eos
    [255, 179, 102],  # con
    [255, 0, 255],  # mitosis
    [0,50,0], # benign
    [50, 0, 0] #macrophage
]



def make_1class_map_thr(mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)+1):
        idx = mask == l
        r[idx] = class_colors [l-1][0]
        g[idx] = class_colors [l-1][1]
        b[idx] = class_colors [l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def _get_ovl_msk_img(img_path, msk_path, color):
    img = np.array(Image.open(img_path))
    msk = np.array(Image.open(msk_path))
    print(np.unique(msk))
    color_map = make_1class_map_thr(msk, color)
    ovl = cv2.addWeighted(img, 0.6, color_map, 0.4, 0)
    return ovl


root = "/home/zhilong/Github_Base/BiomedParse/test_res/"
json_file = ("/media/zhilong/bf844588-a56b-421b-b760-4505fcf10d28/VLM/Dataset/TCGA_LUAD/08_Json_Output/ANNOTATION_L4_s/"
             "TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01_filtered.json")
save_dir = "/home/zhilong/Github_Base/BiomedParse/test_res/biomedparse_res_allcell/"
img_dir = os.path.join(root, "img")
res_dir = os.path.join(root, "res_allcell")

img_lst = os.listdir(img_dir)

for cur_img_n in img_lst:
    cur_img_p = os.path.join(img_dir, cur_img_n)
    cur_msk_p = os.path.join(res_dir, cur_img_n[:-3]+'png')
    cur_ovl = _get_ovl_msk_img(cur_img_p, cur_msk_p, color_cell)
    Image.fromarray(cur_ovl).save(os.path.join(save_dir, cur_img_n[:-3]+'png'))










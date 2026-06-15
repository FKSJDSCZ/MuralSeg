import os
from pathlib import Path

import cv2
import numpy as np
import mmcv
import imgviz
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules

Image.MAX_IMAGE_PIXELS = None

if __name__ == '__main__':
    # build the model from a config file and a checkpoint file
    register_all_modules()

    workdir = Path(r"/home/ubuntu/YWH/MuralSeg")
    save_dir = Path(r"/home/ubuntu/YWH/dataset/dataset0118/label")
    config = workdir / r"local_configs/segformer_mit-b4_4xb4-100k_kizil-512x512.py"
    checkpoint = str(
        workdir / r"runs/trains/segformer_mit-b4_4xb4-100k_kizil-512x512/20260108_033210/best_mIoU_iter_89500.pth"
    )
    device = "cuda:0"
    image_dir = Path(r"/home/ubuntu/YWH/dataset/dataset0118/damaged")

    palette = imgviz.label_colormap()
    palette[0] = [255, 255, 255]
    palette[1] = [0, 0, 0]

    model = init_model(config, checkpoint, device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)

    # result data conversion
    for filename in tqdm(os.listdir(image_dir)):
        image = np.array(Image.open(image_dir / filename).convert('RGB'))
        h, w = image.shape[:2]

        # pad_value = np.mean(image.reshape(-1, 3), axis=0)
        pad_value = [255, 255, 255]
        if w > h:
            padding = (w - h) // 2
            bbox = (padding, padding + h, 0, w)
            image = np.pad(image, ((padding, w - padding - h), (0, 0), (0, 0)), "constant", constant_values=255)
        else:
            padding = (h - w) // 2
            bbox = (0, h, padding, padding + w)
            image = np.pad(image, ((0, 0), (padding, h - padding - w), (0, 0)), "constant", constant_values=255)

        original_h, original_w = image.shape[:2]
        image = np.array(Image.fromarray(image))
        background_mask = np.all(image == (255, 255, 255), axis=-1)
        image[background_mask] = pad_value
        # Image.fromarray(image).save(fr"/home/ubuntu/YWH/MuralSeg/tests/{filename[:-4]}padded.png")
        result = inference_model(model, image)

        pred_np = result.pred_sem_seg.data.cpu().numpy().squeeze(0).astype(np.uint8)
        pred_np[background_mask] = 0
        pred_np = np.where(pred_np <= 1, 0, 1).astype(np.uint8)
        pred_np = np.array(Image.fromarray(pred_np).resize((original_w, original_h), Image.Resampling.NEAREST))
        pred_np = pred_np[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        pred_pil = Image.fromarray(pred_np.astype(np.uint8), mode='P')
        pred_pil.putpalette(palette.flatten())
        pred_pil.save(save_dir / (Path(filename).stem + ".png"))
        # pred_pil.save(fr"/home/ubuntu/YWH/MuralSeg/tests/{filename[:-4]}padded_mask.png")

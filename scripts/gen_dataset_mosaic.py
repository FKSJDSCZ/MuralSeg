import math
import os

import numpy as np
from PIL import Image


def build_mosaic(
    orig_dir,
    index_dir,
    filenames,
    palette,
    alpha=0.5,
    groups_per_col=4,
    gap=10,
    margin=20,
    out_path="mosaic.png",
):
    """
    orig_dir: 原图文件夹
    index_dir: 索引图文件夹
    filenames: 要处理的文件名列表，如 ["1.png", "2.png"]
    palette: 二维数组，形如 [[0,0,0], [255,0,0], [0,255,0], ...]，按 RGB 填写
    alpha: 彩色标注叠加到原图上的透明度，范围 [0, 1]
    groups_per_col: 每列放多少组图像
    gap: 图像与图像之间的距离（组内、组间统一使用）
    margin: 整张大图到边缘的距离
    out_path: 输出大图路径
    """

    if not filenames:
        raise ValueError("filenames 不能为空")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 必须在 [0, 1] 范围内")
    if groups_per_col <= 0:
        raise ValueError("groups_per_col 必须大于 0")

    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette 必须是 shape=(N, 3) 的二维数组")

    group_images = []
    img_h, img_w = None, None

    for name in filenames:
        orig_path = os.path.join(orig_dir, name)
        index_path = os.path.join(index_dir, name)

        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"原图不存在: {orig_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引图不存在: {index_path}")

        # 原图转 RGB，索引图转单通道 L
        orig = np.array(Image.open(orig_path).convert("RGB"), dtype=np.uint8)
        index = np.array(Image.open(index_path).convert("P"), dtype=np.uint8)

        if orig.shape[:2] != index.shape[:2]:
            raise ValueError(f"尺寸不一致: {name}")

        if img_h is None:
            img_h, img_w = orig.shape[:2]
        elif orig.shape[:2] != (img_h, img_w):
            raise ValueError(f"所有图像尺寸必须一致: {name}")

        max_idx = int(index.max())
        if max_idx >= len(palette):
            raise ValueError(
                f"{name} 中最大索引值为 {max_idx}，但 palette 只有 {len(palette)} 种颜色",
            )

        # 索引图 -> 彩色图
        color_mask = palette[index]  # shape: (H, W, 3)

        # alpha 叠加，只叠加 index > 0 的区域
        annotated = orig.copy()
        mask = index > 0
        # mask = np.ones_like(index, dtype=bool)
        annotated[mask] = (
                orig[mask].astype(np.float32) * (1 - alpha)
                + color_mask[mask].astype(np.float32) * alpha
        ).astype(np.uint8)

        group_images.append((orig, annotated))

    n = len(group_images)
    rows = min(groups_per_col, n)
    cols = math.ceil(n / groups_per_col)

    # 每组图像横向排列：原图 | gap | 标注图
    group_w = img_w * 2 + gap
    group_h = img_h

    canvas_h = margin * 2 + rows * group_h + (rows - 1) * gap
    canvas_w = margin * 2 + cols * group_w + (cols - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)  # 白底

    # 按列排：每列 groups_per_col 组，从上到下，再到下一列
    for i, (orig, annotated) in enumerate(group_images):
        col = i // groups_per_col
        row = i % groups_per_col

        x = margin + col * (group_w + gap)
        y = margin + row * (group_h + gap)

        # 原图
        canvas[y:y + img_h, x:x + img_w] = orig

        # 标注图
        x2 = x + img_w + gap
        canvas[y:y + img_h, x2:x2 + img_w] = annotated

    Image.fromarray(canvas).save(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    orig_dir = r"data/kizil/images/train"
    index_dir = r"data/kizil/labels/train"

    filenames = [
        "BackCorridor_LatticePainting_4xDownSampled_2619_3544_1062.png",
        "BackCorridor_NirvanaScene1_4xDownSampled_10000_805_809.png",
        "BackCorridor_NirvanaScene2_4xDownSampled_8274_1304_1801.png",
        "BackCorridor_NirvanaScene3_4xDownSampled_9703_570_1663.png",
        "BackCorridor_Stupa_4xDownSampled_782_736_1402_augmented.png",
        "LeftCorridor_BottomStupa_1321_651_1264_augmented.png",
        "LeftCorridor_BottomStupa_3536_492_1218_augmented.png",
        "LeftCorridor_LatticePainting_998_310_1446.png",
        "LeftCorridor_LatticePainting_1689_3316_1730_augmented.png",
        "LeftCorridor_LatticePainting_2034_198_1240.png",
        "LeftCorridor_LatticePainting_3464_567_1120_augmented.png",
        "LeftCorridor_TopStupa_5468_974_1093.png",
    ]

    # 这里按 RGB 写更直观；脚本内部会自动转成 OpenCV 的 BGR
    palette = [
        [119, 11, 32],  # background
        [107, 142, 35],  # incomplete mural
        [0, 60, 100],  # exposed mud layers
        [0, 0, 142],  # exposed rock layers
        [81, 0, 81],  # protective filling
        [250, 170, 30],  # crack
        [150, 120, 90],  # hole
    ]

    build_mosaic(
        orig_dir=orig_dir,
        index_dir=index_dir,
        filenames=filenames,
        palette=palette,
        alpha=0.4,
        groups_per_col=4,
        gap=16,
        margin=24,
        out_path="vis/kizil_mosaic.png",
    )

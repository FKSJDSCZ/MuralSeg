import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_first_config_value(cfg, keys, default=None):
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def normalize_suffixes(value):
    """Normalize suffix config to a tuple.

    Accepted forms:
      - ".png"
      - "png"  -> ".png"
      - "_mask.png"
      - [".jpg", ".png"]

    If None is returned, the script falls back to IMAGE_EXTS and uses Path.stem
    as the matching key.
    """
    if value is None:
        return None

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise ValueError("Suffix config must be a string or a list/tuple/set of strings.")

    out = []
    for suffix in values:
        if suffix is None:
            continue
        suffix = str(suffix).strip()
        if not suffix:
            continue

        # Common shorthand: "png" means ".png".  Keep custom suffixes such as
        # "_label.png" or "mask.png" unchanged.
        if not suffix.startswith(".") and "." not in suffix and not suffix.startswith("_"):
            suffix = "." + suffix

        out.append(suffix)

    if not out:
        return None

    # Longest first lets "_label.png" win over ".png" if both are configured.
    return tuple(sorted(dict.fromkeys(out), key=len, reverse=True))


def get_suffixes(cfg, *keys):
    value = get_first_config_value(cfg, keys, default=None)
    return normalize_suffixes(value)


def strip_matching_suffix(name, suffixes):
    lower_name = name.lower()
    for suffix in suffixes:
        if lower_name.endswith(suffix.lower()):
            return name[: -len(suffix)]
    return None


def list_image_files(folder, suffixes=None, role="image files"):
    """Return {sample_key: path} for image files in folder.

    When suffixes is None, all IMAGE_EXTS are accepted and the sample key is
    Path(filename).stem.  When suffixes is provided, only filenames ending with
    one of those suffixes are accepted and that exact suffix is removed to build
    the sample key.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{role} folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"{role} path is not a folder: {folder}")

    suffixes = normalize_suffixes(suffixes)
    files = {}

    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue

        if suffixes is None:
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            key = p.stem
        else:
            key = strip_matching_suffix(p.name, suffixes)
            if key is None:
                continue

        if not key:
            raise ValueError(f"Cannot build an empty sample key from file: {p}")

        if key in files:
            raise ValueError(
                f"Multiple {role} map to sample key '{key}': "
                f"'{files[key].name}' and '{p.name}'. "
                "Please specify a more precise suffix.",
            )

        files[key] = p

    return files


def get_model_pred_suffixes(cfg, model_name, model_cfg):
    if isinstance(model_cfg, dict):
        value = get_first_config_value(
            model_cfg,
            ["suffixes", "suffix", "pred_suffixes", "pred_suffix"],
            default=None,
        )
        if value is not None:
            return normalize_suffixes(value)

    per_model = get_first_config_value(
        cfg,
        ["pred_suffix_by_model", "model_pred_suffix_by_model", "model_suffix_by_model"],
        default=None,
    )
    if isinstance(per_model, dict) and model_name in per_model:
        return normalize_suffixes(per_model[model_name])

    # Backward-compatible global names for prediction/model output suffixes.
    value = get_first_config_value(
        cfg,
        ["pred_suffixes", "pred_suffix", "model_pred_suffixes", "model_pred_suffix"],
        default=None,
    )
    return normalize_suffixes(value)


def normalize_model_specs(cfg):
    specs = {}
    for model_name, model_cfg in cfg["models"].items():
        if isinstance(model_cfg, dict):
            pred_dir = get_first_config_value(model_cfg, ["dir", "pred_dir", "path"], default=None)
            if pred_dir is None:
                raise ValueError(
                    f"Model '{model_name}' is configured as a dict, but no "
                    "'dir'/'pred_dir'/'path' field was provided.",
                )
        else:
            pred_dir = model_cfg

        specs[model_name] = {
            "dir": Path(pred_dir),
            "suffixes": get_model_pred_suffixes(cfg, model_name, model_cfg),
        }

    return specs


def collect_file_maps(cfg):
    image_suffixes = get_suffixes(cfg, "image_suffixes", "image_suffix")
    label_suffixes = get_suffixes(cfg, "label_suffixes", "label_suffix")
    model_specs = normalize_model_specs(cfg)

    image_files = list_image_files(cfg["image_dir"], image_suffixes, role="image files")
    label_files = list_image_files(cfg["label_dir"], label_suffixes, role="label files")

    if not image_files:
        raise ValueError(f"No image files found in image_dir: {cfg['image_dir']}")
    if not label_files:
        raise ValueError(f"No label files found in label_dir: {cfg['label_dir']}")

    model_files = {}
    for model_name, spec in model_specs.items():
        pred_files = list_image_files(
            spec["dir"],
            spec["suffixes"],
            role=f"prediction files for model '{model_name}'",
        )
        if not pred_files:
            raise ValueError(f"Prediction folder for model '{model_name}' is empty: {spec['dir']}")
        model_files[model_name] = pred_files

    return {
        "image": image_files,
        "label": label_files,
        "models": model_files,
    }


def build_name_resolver(available_names, *file_maps):
    available_names = set(available_names)
    alias_to_names = {}

    def add_alias(alias, name):
        if not alias or name not in available_names:
            return
        alias_to_names.setdefault(str(alias), set()).add(name)

    for name in available_names:
        add_alias(name, name)
        add_alias(Path(name).stem, name)

    for file_map in file_maps:
        for key, path in file_map.items():
            add_alias(key, key)
            add_alias(path.name, key)
            add_alias(path.stem, key)

    def resolve(requested_name):
        requested_name = str(requested_name)
        hits = alias_to_names.get(requested_name, set())
        if len(hits) == 1:
            return next(iter(hits))
        if len(hits) > 1:
            raise ValueError(f"Requested name '{requested_name}' is ambiguous: {sorted(hits)}")

        stem = Path(requested_name).stem
        hits = alias_to_names.get(stem, set())
        if len(hits) == 1:
            return next(iter(hits))
        if len(hits) > 1:
            raise ValueError(f"Requested name '{requested_name}' is ambiguous: {sorted(hits)}")

        raise FileNotFoundError(
            f"Requested name '{requested_name}' was not found among common sample keys.",
        )

    return resolve


def ensure_rgb(img):
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


def load_font(font_path=None, font_size=24):
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ],
    )
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def get_text_size(text, font):
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def colorize_index_label(index_arr, palette):
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("Palette must be a 2D array with shape [num_classes, 3].")
    max_idx = int(index_arr.max())
    if max_idx >= len(palette):
        raise ValueError(
            f"Label index {max_idx} exceeds palette length {len(palette)}. "
            "Please provide a palette covering all classes.",
        )
    return palette[index_arr]


def make_label_overlay(image_rgb, label_path, alpha, palette=None):
    image_arr = np.array(image_rgb, dtype=np.float32)
    label_img = Image.open(label_path)
    label_arr = np.array(label_img)
    # label_arr[label_arr > 0] -= 1  # label[0] = background

    if label_arr.ndim == 2:
        if palette is None:
            raise ValueError(
                f"Label '{label_path}' is single-channel indexed, but no palette was provided.",
            )
        color_label = colorize_index_label(label_arr, palette)
    elif label_arr.ndim == 3:
        if label_arr.shape[2] == 1:
            if palette is None:
                raise ValueError(
                    f"Label '{label_path}' is single-channel indexed, but no palette was provided.",
                )
            color_label = colorize_index_label(label_arr[..., 0], palette)
        else:
            color_label = label_arr[..., :3].astype(np.uint8)
    else:
        raise ValueError(f"Unsupported label shape: {label_arr.shape}")

    if color_label.shape[:2] != image_arr.shape[:2]:
        color_label = np.array(
            Image.fromarray(color_label).resize(image_rgb.size, resample=RESAMPLE),
            dtype=np.uint8,
        )

    out = (1.0 - alpha) * image_arr + alpha * color_label.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def resize_keep_aspect(img, *, target_height=None, target_width=None):
    w, h = img.size
    if target_height is None and target_width is None:
        raise ValueError("One of target_height or target_width must be provided.")
    if target_height is not None and target_width is not None:
        raise ValueError("Only one of target_height or target_width can be provided.")
    if target_height is not None:
        scale = target_height / h
        new_w = max(1, int(round(w * scale)))
        new_h = int(target_height)
    else:
        scale = target_width / w
        new_w = int(target_width)
        new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=RESAMPLE)


def draw_scaled_box(img, box, ref_size, color=(255, 0, 0), width=3):
    if box is None:
        return img
    x1, y1, x2, y2 = box
    ref_w, ref_h = ref_size
    cur_w, cur_h = img.size
    sx = cur_w / ref_w
    sy = cur_h / ref_h
    scaled_box = [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(scaled_box, outline=color, width=width)
    return out


def prepare_group(filename, cfg, palette, file_maps=None):
    if file_maps is None:
        file_maps = collect_file_maps(cfg)

    image_path = file_maps["image"][filename]
    label_path = file_maps["label"][filename]

    image = ensure_rgb(Image.open(image_path))
    label_overlay = make_label_overlay(
        image_rgb=image,
        label_path=label_path,
        alpha=float(cfg.get("alpha", 0.5)),
        palette=palette,
    )

    preds = {}
    for model_name in cfg["models"].keys():
        pred_path = file_maps["models"][model_name][filename]
        pred_img = ensure_rgb(Image.open(pred_path))
        preds[model_name] = pred_img

    return {
        "filename": filename,
        "Image": image,
        "Ground Truth": label_overlay,
        "preds": preds,
        "orig_size": image.size,
    }


def normalize_boxes(cfg, selected_filenames, file_maps=None):
    if cfg.get("source_mode") != "manual":
        return {name: None for name in selected_filenames}

    boxes = cfg.get("boxes")
    if boxes is None:
        return {name: None for name in selected_filenames}

    if isinstance(boxes, dict):
        out = {}
        for name in selected_filenames:
            candidates = [name, Path(name).stem]

            if file_maps is not None:
                candidate_maps = [file_maps["image"], file_maps["label"]]
                candidate_maps.extend(file_maps["models"].values())
                for file_map in candidate_maps:
                    if name in file_map:
                        path = file_map[name]
                        candidates.extend([path.name, path.stem])

            out[name] = None
            for candidate in candidates:
                if candidate in boxes:
                    out[name] = boxes[candidate]
                    break

        return out

    if isinstance(boxes, list):
        if len(boxes) != len(selected_filenames):
            raise ValueError(
                "When 'boxes' is a list, its length must equal the number of selected filenames.",
            )
        return {name: box for name, box in zip(selected_filenames, boxes)}

    raise ValueError("'boxes' must be either a dict or a list.")


def select_filenames(cfg, file_maps=None):
    if file_maps is None:
        file_maps = collect_file_maps(cfg)

    image_files = file_maps["image"]
    label_files = file_maps["label"]

    common = set(image_files.keys()) & set(label_files.keys())
    for pred_files in file_maps["models"].values():
        common &= set(pred_files.keys())

    common = sorted(common)
    if not common:
        raise ValueError(
            "No common sample keys were found across image/label/model folders. "
            "Check image_suffix, label_suffix, pred_suffix, or per-model suffix settings.",
        )

    source_mode = cfg["source_mode"]
    if source_mode == "random":
        n = int(cfg["num_images"])
        if n > len(common):
            raise ValueError(
                f"Requested num_images={n}, but only {len(common)} common samples are available.",
            )
        seed = cfg.get("seed", None)
        if not seed:
            seed = None
        rng = random.Random(seed)
        selected = sorted(rng.sample(common, n))
    elif source_mode == "manual":
        requested = cfg["filenames"]
        if not requested:
            raise ValueError("'filenames' must be provided in manual source mode.")
        resolver = build_name_resolver(
            common,
            image_files,
            label_files,
            *file_maps["models"].values(),
        )
        selected = [resolver(name) for name in requested]
    else:
        raise ValueError("source_mode must be 'random' or 'manual'.")

    return selected


def paste_center(canvas, img, x0, y0, cell_w, cell_h):
    x = x0 + (cell_w - img.size[0]) // 2
    y = y0 + (cell_h - img.size[1]) // 2
    canvas.paste(img, (x, y))


def draw_text_center(canvas, text, x0, y0, cell_w, cell_h, font):
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (cell_w - tw) // 2
    y = y0 + (cell_h - th) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def render_layout_rows(groups, cfg, boxes_map, font):
    target_h = int(cfg.get("target_height", 256))
    gap = int(cfg.get("gap", 16))
    margin = int(cfg.get("margin", 24))
    text_pad = int(cfg.get("text_padding", 12))
    box_width = int(cfg.get("box_width", 3))

    model_names = list(cfg["models"].keys())
    all_names = ["Ground Truth"] + model_names
    row_names = ["Image"] + all_names
    num_rows = len(row_names)

    scaled = []
    col_widths = []
    for group in groups:
        box = boxes_map.get(group["filename"])
        items = {
            "Image": draw_scaled_box(
                resize_keep_aspect(group["Image"], target_height=target_h),
                box,
                group["orig_size"],
                width=box_width,
            ),
            "Ground Truth": draw_scaled_box(
                resize_keep_aspect(group["Ground Truth"], target_height=target_h),
                box,
                group["orig_size"],
                width=box_width,
            ),
        }
        for model_name in model_names:
            items[model_name] = draw_scaled_box(
                resize_keep_aspect(group["preds"][model_name], target_height=target_h),
                box,
                group["orig_size"],
                width=box_width,
            )
        scaled.append(items)
        col_widths.append(max(items[name].size[0] for name in row_names))

    text_widths = []
    for name in all_names:
        tw, _ = get_text_size(name, font)
        text_widths.append(tw)
    text_col_w = max([0] + text_widths) + 2 * text_pad

    canvas_w = 2 * margin + sum(col_widths) + text_col_w + gap * (len(groups))
    canvas_h = 2 * margin + num_rows * target_h + gap * (num_rows - 1)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    x_positions = []
    x = margin
    for w in col_widths:
        x_positions.append(x)
        x += w + gap
    text_x = x

    y_positions = [margin + i * (target_h + gap) for i in range(num_rows)]

    for col_idx, items in enumerate(scaled):
        for row_idx, row_name in enumerate(row_names):
            paste_center(
                canvas,
                items[row_name],
                x_positions[col_idx],
                y_positions[row_idx],
                col_widths[col_idx],
                target_h,
            )

    for row_idx, model_name in enumerate(all_names, start=1):
        draw_text_center(
            canvas,
            model_name,
            text_x,
            y_positions[row_idx],
            text_col_w,
            target_h,
            font,
        )

    return canvas


def render_layout_cols(groups, cfg, boxes_map, font):
    target_w = int(cfg.get("target_width", 256))
    gap = int(cfg.get("gap", 16))
    margin = int(cfg.get("margin", 24))
    text_pad = int(cfg.get("text_padding", 12))
    box_width = int(cfg.get("box_width", 3))

    model_names = list(cfg["models"].keys())
    all_names = ["Ground Truth"] + model_names
    col_names = ["Image"] + all_names

    scaled_rows = []
    row_heights = []
    for group in groups:
        box = boxes_map.get(group["filename"])
        items = {
            "Image": draw_scaled_box(
                resize_keep_aspect(group["Image"], target_width=target_w),
                box,
                group["orig_size"],
                width=box_width,
            ),
            "Ground Truth": draw_scaled_box(
                resize_keep_aspect(group["Ground Truth"], target_width=target_w),
                box,
                group["orig_size"],
                width=box_width,
            ),
        }
        for model_name in model_names:
            items[model_name] = draw_scaled_box(
                resize_keep_aspect(group["preds"][model_name], target_width=target_w),
                box,
                group["orig_size"],
                width=box_width,
            )
        scaled_rows.append(items)
        row_heights.append(max(items[name].size[1] for name in col_names))

    col_widths = [target_w, target_w]
    for model_name in all_names:
        tw, _ = get_text_size(model_name, font)
        col_widths.append(max(target_w, tw + 2 * text_pad))

    text_h = max([get_text_size(name, font)[1] for name in all_names] + [font.size]) + 2 * text_pad

    canvas_w = 2 * margin + sum(col_widths) + gap * (len(col_widths) - 1)
    canvas_h = 2 * margin + sum(row_heights) + text_h + gap * (len(row_heights))

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    x_positions = []
    x = margin
    for w in col_widths:
        x_positions.append(x)
        x += w + gap

    y_positions = []
    y = margin
    for h in row_heights:
        y_positions.append(y)
        y += h + gap
    text_y = y

    for row_idx, items in enumerate(scaled_rows):
        for col_idx, col_name in enumerate(col_names):
            paste_center(
                canvas,
                items[col_name],
                x_positions[col_idx],
                y_positions[row_idx],
                col_widths[col_idx],
                row_heights[row_idx],
            )

    for col_idx, model_name in enumerate(all_names, start=1):
        draw_text_center(
            canvas,
            model_name,
            x_positions[col_idx],
            text_y,
            col_widths[col_idx],
            text_h,
            font,
        )

    return canvas


def save_selected_names(output_path, selected_filenames):
    output_path = Path(output_path)
    txt_path = output_path.with_suffix(output_path.suffix + ".selected_filenames.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name in selected_filenames:
            f.write(name + "\n")
    return txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a CV paper-style comparison figure by stitching original images, GT overlays, and model predictions.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    palette = cfg.get("palette", None)
    font = load_font(cfg.get("font_path"), int(cfg.get("font_size", 24)))

    file_maps = collect_file_maps(cfg)
    selected_filenames = select_filenames(cfg, file_maps)
    boxes_map = normalize_boxes(cfg, selected_filenames, file_maps)

    groups = [prepare_group(name, cfg, palette, file_maps) for name in selected_filenames]

    layout_mode = cfg["layout_mode"]
    if layout_mode == "rows":
        canvas = render_layout_rows(groups, cfg, boxes_map, font)
    elif layout_mode == "cols":
        canvas = render_layout_cols(groups, cfg, boxes_map, font)
    else:
        raise ValueError("layout_mode must be 'rows' or 'cols'.")

    output_path = cfg["output_path"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    txt_path = save_selected_names(output_path, selected_filenames)

    print("Done.")
    print(f"Saved figure to: {output_path}")
    print(f"Saved selected filenames to: {txt_path}")
    print("Selected sample keys:")
    for name in selected_filenames:
        print(name)


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "/Users/huynhthithanhvi/Projects/FashionERN/datasets/shoes_dataset"
)

OUTPUT_DIR = DATASET_ROOT / "shoes_local_feature_13"

CLIP_MODEL_NAME = "RN50x4"

IMAGE_SIZE = 288
TARGET_RATIO = 1.25

# RN50x4 feature dimension used by FashionERN
FEATURE_DIM = 640

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size for 13 patches
BATCH_SIZE = 16


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    """
    Same TargetPad logic used by FashionERN.
    """

    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size

        actual_ratio = max(w, h) / min(w, h)

        if actual_ratio < self.target_ratio:
            return image

        scaled_max_wh = max(w, h) / self.target_ratio

        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)

        padding = [hp, vp, hp, vp]

        return TF.pad(
            image,
            padding,
            0,
            "constant",
        )


def targetpad_transform(
    target_ratio=1.25,
    dim=288,
):
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(
            dim,
            interpolation=InterpolationMode.BICUBIC,
        ),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ============================================================
# PATCH GENERATION
# ============================================================

def cut_image_4(image):
    """
    2 x 2 grid = 4 patches
    """

    width, height = image.size

    item_width = width // 2
    item_height = height // 2

    boxes = []

    for i in range(2):
        for j in range(2):

            box = (
                j * item_width,
                i * item_height,
                (j + 1) * item_width,
                (i + 1) * item_height,
            )

            boxes.append(box)

    return [image.crop(box) for box in boxes]


def cut_image_9(image):
    width, height = image.size

    item_width = width // 3
    item_height = height // 3

    boxes = []

    for i in range(3):
        for j in range(3):

            box = (
                j * item_width,
                i * item_height,
                (j + 1) * item_width,
                (i + 1) * item_height,
            )

            boxes.append(box)

    return [image.crop(box) for box in boxes]


def create_13_patches(image):
    """
    4 patches + 9 patches = 13 patches
    """

    patches_4 = cut_image_4(image)
    patches_9 = cut_image_9(image)

    patches = patches_4 + patches_9

    assert len(patches) == 13

    return patches


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(root):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    }

    images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        # Bỏ qua symbolic links
        if path.is_symlink():
            continue

        # Không lấy feature output làm input
        if path.parent == OUTPUT_DIR:
            continue

        if path.suffix in extensions:
            images.append(path)

    return sorted(images)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FashionERN Shoes Local Feature Extraction")
    print("=" * 60)

    print(f"Dataset : {DATASET_ROOT}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Model   : {CLIP_MODEL_NAME}")
    print(f"Device  : {DEVICE}")
    print(f"Feature : [{13}, {FEATURE_DIM}]")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    print("\nLoading CLIP RN50x4...")

    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        device=DEVICE,
    )

    model.eval()
    model.float()

    print("CLIP loaded successfully.")

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print("\nSearching Shoes images...")

    image_paths = find_images(DATASET_ROOT)

    print(f"Images found: {len(image_paths)}")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    preprocess = targetpad_transform(
        TARGET_RATIO,
        IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0

    with torch.no_grad():

        for image_path in tqdm(
            image_paths,
            desc="Extracting",
        ):

            filename = image_path.name

            output_path = OUTPUT_DIR / (
                filename + ".pth"
            )

            # ----------------------------------------------
            # Skip existing
            # ----------------------------------------------

            if output_path.exists():

                skipped += 1
                continue

            try:

                # ------------------------------------------
                # Open image
                # ------------------------------------------

                image = Image.open(
                    image_path
                ).convert("RGB")

                # ------------------------------------------
                # Resize original before patching
                # ------------------------------------------

                image = image.resize(
                    (360, 360),
                    Image.Resampling.LANCZOS,
                )

                # ------------------------------------------
                # Generate 13 patches
                # ------------------------------------------

                patches = create_13_patches(
                    image
                )

                # ------------------------------------------
                # Preprocess patches
                # ------------------------------------------

                patch_tensors = torch.stack(
                    [
                        preprocess(patch)
                        for patch in patches
                    ]
                )

                # ------------------------------------------
                # CLIP inference
                # ------------------------------------------

                features = []

                for start in range(
                    0,
                    len(patch_tensors),
                    BATCH_SIZE,
                ):

                    batch = patch_tensors[
                        start:start + BATCH_SIZE
                    ].to(
                        DEVICE,
                        non_blocking=True,
                    )

                    batch_features = model.encode_image(
                        batch
                    )

                    batch_features = batch_features.float()

                    features.append(
                        batch_features.cpu()
                    )

                feature_all = torch.cat(
                    features,
                    dim=0,
                )

                # ------------------------------------------
                # Verify shape
                # ------------------------------------------

                if feature_all.shape != (
                    13,
                    FEATURE_DIM,
                ):

                    raise RuntimeError(
                        f"Unexpected feature shape "
                        f"{feature_all.shape} "
                        f"for {filename}"
                    )

                # ------------------------------------------
                # Save
                # ------------------------------------------

                torch.save(
                    feature_all,
                    output_path,
                )

                processed += 1

            except Exception as e:

                failed += 1

                print(
                    f"\nERROR: {image_path}"
                )

                print(e)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Images found     : {len(image_paths)}"
    )

    print(
        f"New features     : {processed}"
    )

    print(
        f"Already existed  : {skipped}"
    )

    print(
        f"Failed           : {failed}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

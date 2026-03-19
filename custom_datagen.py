import os
import re
import numpy as np


def load_img(img_dir, img_list):
    """
    Load .npy files from img_list and stack them into a numpy array of shape (B, ...).

    The function interface remains unchanged.
    """
    images = []
    for image_name in img_list:
        if isinstance(image_name, str) and image_name.endswith(".npy"):
            full_path = os.path.join(img_dir, image_name)
            image = np.load(full_path)
            images.append(image.astype(np.float32))

    return np.asarray(images, dtype=np.float32)


def _extract_id(filename: str):
    """
    Extract the numeric ID at the end of the filename.

    Example:
        image_0.npy -> 0
        mask_0.npy  -> 0

    Returns:
        int or None if no match is found.
    """
    m = re.search(r"_(\d+)\.npy$", filename)
    return int(m.group(1)) if m else None


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    Infinite generator that yields batches of (X, Y).

    Fixes and improvements:
    1) Pair images and masks using ID:
       image_{id}.npy <-> mask_{id}.npy
    2) Avoid mismatching caused by unordered os.listdir()
    3) Use os.path.join for cross-platform compatibility
    4) Optional shuffling controlled via environment variable:
       DATAGEN_SHUFFLE=1 (no interface change required)
    """

    # ---- 1) Filter only .npy files ----
    img_files = [f for f in img_list if isinstance(f, str) and f.endswith(".npy")]
    mask_files = [f for f in mask_list if isinstance(f, str) and f.endswith(".npy")]

    # ---- 2) Build mapping: id -> filename ----
    img_map = {}
    for f in img_files:
        fid = _extract_id(f)
        if fid is not None:
            img_map[fid] = f

    mask_map = {}
    for f in mask_files:
        fid = _extract_id(f)
        if fid is not None:
            mask_map[fid] = f

    common_ids = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if len(common_ids) == 0:
        raise ValueError(
            "No matching image/mask pairs found (based on _{id}.npy rule).\n"
            "Expected format example: images/image_0.npy and masks/mask_0.npy"
        )

    paired_imgs = [img_map[i] for i in common_ids]
    paired_masks = [mask_map[i] for i in common_ids]
    L = len(common_ids)

    # ---- 3) Optional shuffle (controlled via environment variables) ----
    shuffle = os.getenv("DATAGEN_SHUFFLE", "0") == "1"
    seed = int(os.getenv("DATAGEN_SEED", "1337"))
    rng = np.random.default_rng(seed)
    indices = np.arange(L)

    while True:
        if shuffle:
            rng.shuffle(indices)

        batch_start = 0
        while batch_start < L:
            batch_end = min(batch_start + batch_size, L)
            batch_idx = indices[batch_start:batch_end]

            batch_img_names = [paired_imgs[i] for i in batch_idx]
            batch_mask_names = [paired_masks[i] for i in batch_idx]

            X = load_img(img_dir, batch_img_names)
            Y = load_img(mask_dir, batch_mask_names)

            # Store filenames of the current batch (useful for debugging or logging)
            imageLoader.last_batch_img_names = batch_img_names
            imageLoader.last_batch_mask_names = batch_mask_names

            # Keep masks unchanged (may be label maps or one-hot encoded)
            yield (X, Y)

            batch_start += batch_size
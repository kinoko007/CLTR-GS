# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from PIL import Image
import torch
from utils.graphics_utils import fov2focal

WARNED = False


def PILtoTorch(img, resolution):
    if isinstance(img, Image.Image):
        im = img
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            im = Image.fromarray(img.astype(np.uint8), mode="L")
        elif img.ndim == 3:
            if img.shape[2] == 1:
                im = Image.fromarray(img[..., 0].astype(np.uint8), mode="L")
            elif img.shape[2] == 3:
                im = Image.fromarray(img.astype(np.uint8), mode="RGB")
            elif img.shape[2] == 4:
                im = Image.fromarray(img.astype(np.uint8), mode="RGBA")
            else:
                raise ValueError(f"Unsupported channels in numpy image: {img.shape}")
        else:
            raise TypeError(f"Unsupported numpy image shape: {img.shape}")
    else:
        raise TypeError(f"Unsupported type for PILtoTorch: {type(img)}")

    target_w, target_h = resolution

    single_channel_modes = {"L", "1", "I", "F"}
    if im.mode in single_channel_modes:
        if im.size != (target_w, target_h):
            im = im.resize((target_w, target_h), Image.LANCZOS)
        arr = np.asarray(im)  # HxW
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        t = torch.from_numpy(arr)[None, ...].contiguous()  # 1xHxW
        return t

    if im.mode == "RGBA":
        im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    if im.size != (target_w, target_h):
        im = im.resize((target_w, target_h), Image.LANCZOS)

    arr = np.asarray(im)  # HxWx3
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3xHxW
    return t


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    # Decide target resolution
    if args.resolution in [1, 2, 4, 8]:
        target_w = round(orig_w / (resolution_scale * args.resolution))
        target_h = round(orig_h / (resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600.0
            else:
                global_down = 1.0
        else:
            global_down = orig_w / float(args.resolution)

        scale = float(global_down) * float(resolution_scale)
        target_w = int(orig_w / scale)
        target_h = int(orig_h / scale)

    resolution = (target_w, target_h)

    # Build intrinsics K from FoV and new resolution
    fx = fov2focal(cam_info.FovX, target_w)
    fy = fov2focal(cam_info.FovY, target_h)
    K = np.array([
        [fx, 0.0, target_w / 2.0],
        [0.0, fy, target_h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Load image and optional alpha mask
    if len(cam_info.image.split()) > 3:
        r, g, b, a = cam_info.image.split()
        resized_image_rgb = torch.cat(
            [PILtoTorch(r, resolution), PILtoTorch(g, resolution), PILtoTorch(b, resolution)],
            dim=0
        )
        loaded_mask = PILtoTorch(a, resolution)  # 1xHxW
        gt_image = resized_image_rgb
    else:
        gt_image = PILtoTorch(cam_info.image, resolution)  # 3xHxW
        loaded_mask = None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        K=K,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

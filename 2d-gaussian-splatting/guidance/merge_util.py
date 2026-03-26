import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))

from utils.general_utils import seed_everything
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import shutil
import json
from glob import glob


def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")


def _list_png_files_only(folder):
    """Return sorted list of png filenames (not directories) in folder."""
    if (folder is None) or (not os.path.isdir(folder)):
        return []
    out = []
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp) and f.lower().endswith(".png"):
            out.append(f)
    return sorted(out)


def _list_warp_frames(warp_root_dir):
    """
    Return sorted list of warp_frame filenames and their indices:
      warp_frame000123.png -> 123
    """
    warp_paths = glob(os.path.join(warp_root_dir, "warp_frame*.png"))
    warp_names = [os.path.basename(p) for p in warp_paths]
    warp_names = sorted(warp_names)

    indices = []
    for name in warp_names:
        stem = os.path.splitext(name)[0]  # warp_frame000123
        # extract last 6 digits
        try:
            idx = int(stem[-6:])
        except Exception:
            continue
        indices.append(idx)

    # keep consistent ordering by idx
    pairs = sorted(zip(indices, warp_names), key=lambda x: x[0])
    indices = [p[0] for p in pairs]
    warp_names = [p[1] for p in pairs]
    return indices, warp_names


def replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir):
    """
    Replace visible region (mask==255) by GS render, invisible by inpaint.
    Assumes:
      warp_root_dir: warp_frameXXXXXX.png, mask_frameXXXXXX.png
      inpaint_root_dir: predict_warp_frameXXXXXX.png
    Outputs:
      save_root_dir: predict_warp_frameXXXXXX.png
    """
    os.makedirs(save_root_dir, exist_ok=True)

    if not os.path.isdir(warp_root_dir):
        raise NotADirectoryError(f"[merge_util] warp_root_dir not found: {warp_root_dir}")
    if not os.path.isdir(inpaint_root_dir):
        raise NotADirectoryError(f"[merge_util] inpaint_root_dir not found: {inpaint_root_dir}")

    indices, _ = _list_warp_frames(warp_root_dir)
    if len(indices) == 0:
        raise RuntimeError(
            f"[merge_util] No warp_frame*.png found in {warp_root_dir}\n"
            "  Expect files like warp_frame000000.png"
        )

    for idx in indices:
        gs_render_img_path = os.path.join(warp_root_dir, f'warp_frame{idx:06d}.png')
        mask_img_path = os.path.join(warp_root_dir, f'mask_frame{idx:06d}.png')
        inpaint_img_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{idx:06d}.png')

        if not os.path.exists(gs_render_img_path):
            raise FileNotFoundError(f"[merge_util] Missing GS render: {gs_render_img_path}")
        if not os.path.exists(mask_img_path):
            raise FileNotFoundError(f"[merge_util] Missing mask: {mask_img_path}")
        if not os.path.exists(inpaint_img_path):
            raise FileNotFoundError(f"[merge_util] Missing inpaint result: {inpaint_img_path}")

        gs_render_img = Image.open(gs_render_img_path).convert("RGB")
        mask_img = Image.open(mask_img_path).convert("L")
        inpaint_img = Image.open(inpaint_img_path).convert("RGB")

        mask_map = (np.array(mask_img) / 255.0).astype(np.float32)  # 0/1
        gs_render_np = np.array(gs_render_img, dtype=np.uint8)
        inpaint_np = np.array(inpaint_img, dtype=np.uint8)

        # visible part is gs_render_img, invisible part is inpaint_img
        save_np = inpaint_np.copy()
        save_np[mask_map == 1] = gs_render_np[mask_map == 1]

        Image.fromarray(save_np).save(os.path.join(save_root_dir, f'predict_warp_frame{idx:06d}.png'))
        print(f'Inpaint {idx:06d} replace done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--plane_root_dir', type=str, required=True)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--none_replace", action='store_true')
    parser.add_argument("--anchor_view_id_json_path", type=str, required=True)
    args = parser.parse_args()

    seed_everything()

    see3d_root_dir = os.path.join(args.source_path, 'see3d_render')
    cur_see3d_root_dir = os.path.join(see3d_root_dir, f'stage{args.see3d_stage}')
    inpaint_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-inpainted')
    save_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-inpainted-merged')

    warp_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs')

    # 1) replace inpaint results
    if not args.none_replace:
        replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir)
        print(f'See3D stage {args.see3d_stage} replace inpaint results done!')

    # 2) copy inpaint results to all inpaint folder
    all_inpaint_image_dir = os.path.join(see3d_root_dir, 'inpainted_images')
    os.makedirs(all_inpaint_image_dir, exist_ok=True)

    # begin_idx should count only existing png files (ignore folders like .ipynb_checkpoints)
    begin_idx = len(_list_png_files_only(all_inpaint_image_dir))

    cur_result_dir = save_root_dir if not args.none_replace else inpaint_root_dir
    if not os.path.isdir(cur_result_dir):
        raise NotADirectoryError(f"[merge_util] cur_result_dir not found: {cur_result_dir}")

    # Only copy predict_*.png files, skip directories and other pngs (debug/cat_img/etc)
    candidates = []
    for f in os.listdir(cur_result_dir):
        fp = os.path.join(cur_result_dir, f)
        if not os.path.isfile(fp):
            continue
        if not f.lower().endswith(".png"):
            continue
        if not f.startswith("predict_"):
            continue
        candidates.append(f)
    candidates = sorted(candidates)

    if len(candidates) == 0:
        raise RuntimeError(
            f"[merge_util] No predict_*.png found in {cur_result_dir}\n"
            "  Expect files like predict_warp_frame000000.png"
        )

    for result_img_name in candidates:
        src = os.path.join(cur_result_dir, result_img_name)
        dst = os.path.join(all_inpaint_image_dir, f'predict_warp_frame{begin_idx:06d}.png')
        shutil.copy(src, dst)
        begin_idx += 1

    print(f'See3D stage {args.see3d_stage} copy inpaint results to all inpaint folder done!')

    # 3) merge novel cameras
    see3d_cam_path = os.path.join(see3d_root_dir, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        pre_see3d_cameras = dict(np.load(see3d_cam_path))
        pre_see3d_views = int(pre_see3d_cameras['n_views'])
        os.remove(see3d_cam_path)
    else:
        pre_see3d_cameras = {}
        pre_see3d_views = 0

    cur_see3d_cam_path = os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_see3d_cameras.npz')
    if not os.path.exists(cur_see3d_cam_path):
        raise FileNotFoundError(f"[merge_util] Missing stage cameras npz: {cur_see3d_cam_path}")

    cur_see3d_cameras = np.load(cur_see3d_cam_path)
    cur_see3d_views = int(cur_see3d_cameras['n_views'])

    for i in range(cur_see3d_views):
        cur_id = i + pre_see3d_views
        pre_see3d_cameras[f'R_{cur_id:06d}'] = cur_see3d_cameras[f'R_{i:06d}']
        pre_see3d_cameras[f'T_{cur_id:06d}'] = cur_see3d_cameras[f'T_{i:06d}']
        pre_see3d_cameras[f'FoVx_{cur_id:06d}'] = cur_see3d_cameras[f'FoVx_{i:06d}']
        pre_see3d_cameras[f'FoVy_{cur_id:06d}'] = cur_see3d_cameras[f'FoVy_{i:06d}']
        pre_see3d_cameras[f'image_width_{cur_id:06d}'] = cur_see3d_cameras[f'image_width_{i:06d}']
        pre_see3d_cameras[f'image_height_{cur_id:06d}'] = cur_see3d_cameras[f'image_height_{i:06d}']

    pre_see3d_cameras['n_views'] = cur_see3d_views + pre_see3d_views
    if 'train_views' not in pre_see3d_cameras:
        pre_see3d_cameras['train_views'] = cur_see3d_cameras['train_views']

    np.savez(see3d_cam_path, **pre_see3d_cameras)
    print(f'See3D stage {args.see3d_stage} merge novel cameras done!')

    # 4) merge geometry cues
    plane_root_dir = args.plane_root_dir
    cur_plane_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-planes')

    os.makedirs(plane_root_dir, exist_ok=True)
    if not os.path.isdir(cur_plane_root_dir):
        raise NotADirectoryError(f"[merge_util] Missing current plane dir: {cur_plane_root_dir}")

    plane_file_list = os.listdir(plane_root_dir)
    plane_rgb_list = [file for file in plane_file_list if 'rgb_frame' in file]
    begin_plane_idx = len(plane_rgb_list)

    anchor_view_id_list = []
    for i in range(cur_see3d_views):
        # rgb
        shutil.copy(os.path.join(cur_plane_root_dir, f'rgb_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'rgb_frame{begin_plane_idx:06d}.png'))

        # depth
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.tiff'),
                    os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.tiff'),
                    os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.png'))

        # normal
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.npy'),
                    os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.npy'),
                    os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.npy'),
                    os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.png'))

        # visibility
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.npy'),
                    os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.png'))

        # 2D plane
        shutil.copy(os.path.join(cur_plane_root_dir, f'plane_mask_frame{i:06d}.npy'),
                    os.path.join(plane_root_dir, f'plane_mask_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'plane_vis_frame{i:06d}.png'),
                    os.path.join(plane_root_dir, f'plane_vis_frame{begin_plane_idx:06d}.png'))

        anchor_view_id_list.append(begin_plane_idx)
        begin_plane_idx += 1

    # copy need inpaint views points
    src_ply = os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply')
    if os.path.exists(src_ply):
        shutil.copy(src_ply, os.path.join(plane_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply'))

    with open(args.anchor_view_id_json_path, 'w') as f:
        json.dump(anchor_view_id_list, f)

    print(f'See3D stage {args.see3d_stage} merge geometry cues done!')

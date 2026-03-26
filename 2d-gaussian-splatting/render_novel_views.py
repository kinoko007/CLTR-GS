import os
import sys
sys.path.append(os.getcwd())

import math
import shutil
import numpy as np
import torch
from argparse import ArgumentParser
from PIL import Image
import trimesh

from scene import Scene, GaussianModel
from scene.dataset_readers import load_see3d_cameras
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.render_utils import save_img_f32, save_img_u8
from utils.general_utils import safe_state

from guidance.cam_utils import (
    generate_see3d_camera_by_lookat,
    select_need_inpaint_views,
    generate_see3d_camera_by_lookat_object_centric,
    generate_see3d_camera_by_view_angle,
    generate_see3d_camera_by_lookat_all_plane,
    MiniCam,
)

from matcha.dm_scene.charts import depths_to_points_parallel

from guidance.vis_grid import VisibilityGrid
from planes.get_global_3Dpnts import (
    get_visible_mask_for_input_views,
    get_all_global_3Dpnts
)

# Global Settings
EMPTY_MASK_THRESHOLD = 0.001  # 0.1%

# Curriculum control
BASE_TOP_BAND_RATIO = 0.25
BASE_TOP_MISSING_THRESH = 0.05  # base
CURR_MASK_BLACK_RATIO_THRESH = 0.05  # curriculum mask_frame

# Small utils
def _cuda_release(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    torch.cuda.empty_cache()

# Raw-GS cleanup
def cleanup_unselected_raw_gs(gs_output_dir, total_views, keep_ids):
    keep = set(int(x) for x in keep_ids)
    patterns = [
        "ori_warp_frame{vid:06d}.png",
        "depth_frame{vid:06d}.tiff",
        "alpha_{vid:06d}.npy",
        "alpha_mask_frame{vid:06d}.png",
        "alpha_warp_frame{vid:06d}.png",
        "warp_frame{vid:06d}.png",
        "mask_frame{vid:06d}.png",
    ]

    removed = 0
    for vid in range(int(total_views)):
        if vid in keep:
            continue
        for p in patterns:
            fp = os.path.join(gs_output_dir, p.format(vid=vid))
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    removed += 1
                except Exception as e:
                    print(f"[Cleanup][Warn] Failed to remove {fp}: {e}")
    print(f"[Cleanup] raw-gs cleaned: removed_files={removed}, kept_views={len(keep)}, total_views={total_views}")

# Alpha-warp fallback helpers
def _is_all_black_rgb_u8(img_u8, thr=0):
    if img_u8 is None:
        return True
    return (img_u8.max() <= thr)

def _alpha_to_rgb_u8(alpha_float01):
    a = np.clip(alpha_float01, 0.0, 1.0)
    a_u8 = (a * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(a_u8[..., None], 3, axis=2)

def _load_u8_mask(path):
    m = Image.open(path).convert("L")
    return np.array(m, dtype=np.uint8)

def _top_missing_from_u8_mask(mask_u8, band_ratio=0.25, on_thr=127):
    if mask_u8 is None or mask_u8.size == 0:
        return 1.0
    H = mask_u8.shape[0]
    top_h = max(1, int(H * band_ratio))
    top_band = mask_u8[:top_h, :]
    vis = (top_band > on_thr).astype(np.float32).mean()
    return float(1.0 - vis)

def _mask_black_ratio_u8(mask_u8, on_thr=127):
    if mask_u8 is None or mask_u8.size == 0:
        return 1.0
    visible = (mask_u8 > on_thr).astype(np.float32).mean()
    return float(1.0 - visible)

# Curriculum helpers
def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float32)

def _rodrigues(axis, angle_rad):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = _skew(axis.astype(np.float32))
    I = np.eye(3, dtype=np.float32)
    return I + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def generate_curriculum_lr_cams(base_cam: MiniCam, yaw_deg: float):
    """
    Generate curriculum cameras around base_cam:
      - left/right: yaw around camera up axis (R[:,1])
    Keep camera center fixed.
    """
    R = _to_numpy(base_cam.R).astype(np.float32)              # c2w
    T = _to_numpy(base_cam.T).astype(np.float32).reshape(3)   # w2c
    C = -(R @ T)                                              # camera center (world)

    up_axis = R[:, 1]
    yaw = math.radians(float(yaw_deg))

    def make_cam(R_new):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_new
        c2w[:3, 3] = C
        return MiniCam(
            c2w,
            width=base_cam.image_width,
            height=base_cam.image_height,
            fovy=base_cam.FoVy,
            fovx=base_cam.FoVx,
            znear=getattr(base_cam, "znear", 0.01),
            zfar=getattr(base_cam, "zfar", 100.0),
        )

    outs = []
    if float(yaw_deg) > 0:
        R_left  = _rodrigues(up_axis, -yaw) @ R
        R_right = _rodrigues(up_axis, +yaw) @ R
        outs.append(("left", make_cam(R_left)))
        outs.append(("right", make_cam(R_right)))
    return outs

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--select_inpaint_num", required=True, type=str)
    parser.add_argument("--curriculum_rot_deg", type=float, default=None,
                        help="Yaw rotation degree for curriculum (left/right).")
    parser.add_argument("--alpha_vis_thresh", type=float, default=0.99,
                        help="Alpha threshold for visibility masking.")
    parser.add_argument("--no_delete_unselected_views",
                        dest="delete_unselected_views",
                        action="store_false",
                        help="Do NOT delete unselected novel-view files under raw-gs/ (no inpaint needed).")
    parser.set_defaults(delete_unselected_views=True)

    args = get_combined_args(parser)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_viewpoints = scene.getTrainCameras().copy()
    input_view_num = len(train_viewpoints)

    see3d_render_path = os.path.join(args.source_path, 'see3d_render')
    os.makedirs(see3d_render_path, exist_ok=True)

    # Copy reference images
    ref_views_save_root_path = os.path.join(see3d_render_path, 'ref-views')
    if not os.path.exists(ref_views_save_root_path):
        os.makedirs(ref_views_save_root_path, exist_ok=True)
        src_image_root_path = os.path.join(args.source_path, 'images')
        temp_image_name = os.listdir(src_image_root_path)[0]
        postfix = temp_image_name.split('.')[-1]
        for viewpoint in train_viewpoints:
            image_name = f'{viewpoint.image_name}.{postfix}'
            shutil.copy(os.path.join(src_image_root_path, image_name),
                        os.path.join(ref_views_save_root_path, image_name))

    # Load see3d cameras
    see3d_cam_path = os.path.join(see3d_render_path, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        see3d_viewpoints, _ = load_see3d_cameras(see3d_cam_path, os.path.join(see3d_render_path, 'inpainted_images'))
        train_viewpoints.extend(see3d_viewpoints)
        print(f'Stage {args.see3d_stage} See3D cameras loaded from {see3d_cam_path}')
    else:
        if args.see3d_stage > 1:
            raise AssertionError('See3D cameras not found, but see3d_stage > 1')

    novel_views_save_root_path = os.path.join(see3d_render_path, f'stage{args.see3d_stage}')
    os.makedirs(novel_views_save_root_path, exist_ok=True)
    alpha_vis_thresh = float(args.alpha_vis_thresh)

    # Render train views
    train_save_root_path = os.path.join(novel_views_save_root_path, 'render-train-views')
    os.makedirs(train_save_root_path, exist_ok=True)
    train_view_depths = []
    for train_viewpoint in train_viewpoints:
        idx = train_viewpoint.colmap_id
        render_pkg = render(train_viewpoint, gaussians, pipe, background)
        rgb, depth = render_pkg['render'], render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach().cpu().numpy())
        save_img_u8(rgb.permute(1, 2, 0).detach().cpu().numpy(), os.path.join(train_save_root_path, f'{idx:05d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'depth_{idx:05d}.tiff'))
        _cuda_release(render_pkg, rgb, depth)
    print('Train views render done!')

    input_view_depths = train_view_depths[:input_view_num]
    input_viewpoints = train_viewpoints[:input_view_num]
    gs_input_view_depths = torch.from_numpy(np.array(input_view_depths)).cuda().unsqueeze(1)
    gs_input_view_points = depths_to_points_parallel(gs_input_view_depths, input_viewpoints)

    bbox_min = torch.min(gaussians.get_xyz, dim=0).values
    bbox_max = torch.max(gaussians.get_xyz, dim=0).values
    visibility_grid = VisibilityGrid(bbox_min, bbox_max, 256, train_viewpoints, train_view_depths)
    visibility_grid.vis_invisible_pnts(os.path.join(novel_views_save_root_path, 'invisible_points.ply'))

    novel_poses, novel_cams = [], []
    plane_root_path = os.path.join(args.source_path, 'plane-refine-depths')
    vis_plane_pnts_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_vis_global_3Dplane_points')

    if args.see3d_stage == 1:
        used_fov_deg, only_warp_input_views, select_view_method, used_top_k = 80, False, 'covisibility_rate', 5
        poses1, cams1 = generate_see3d_camera_by_lookat_object_centric(train_viewpoints, visibility_grid, n_frames=40, fovy_deg=used_fov_deg)
        novel_poses.extend(poses1); novel_cams.extend(cams1)
        poses2, cams2 = generate_see3d_camera_by_lookat(input_viewpoints, visibility_grid, gs_input_view_depths.squeeze(1), gs_input_view_points, n_frames=40, fovy_deg=used_fov_deg)
        novel_poses.extend(poses2); novel_cams.extend(cams2)
    elif args.see3d_stage == 2:
        used_fov_deg, only_warp_input_views, select_view_method, used_top_k = 80, False, 'covisibility_rate', 5
        poses1, cams1 = generate_see3d_camera_by_view_angle(input_viewpoints, visibility_grid, fovy_deg=used_fov_deg, n_frames=60)
        novel_poses.extend(poses1); novel_cams.extend(cams1)
    elif args.see3d_stage == 3:
        used_fov_deg, only_warp_input_views, select_view_method, used_top_k = 100, True, 'none_visible_rate', 10
    else:
        raise ValueError(f'Invalid see3d_stage: {args.see3d_stage}')

    plane_all_points_dict = get_all_global_3Dpnts(args.source_path, plane_root_path, see3d_render_path, vis_plane_pnts_path, top_k=used_top_k)
    poses3, cams3 = generate_see3d_camera_by_lookat_all_plane(train_viewpoints, visibility_grid, plane_all_points_dict, fovy_deg=used_fov_deg)
    novel_poses.extend(poses3); novel_cams.extend(cams3)

    # Render GS
    gs_output_dir = os.path.join(novel_views_save_root_path, 'raw-gs')
    os.makedirs(gs_output_dir, exist_ok=True)
    gs_depths, none_visible_rate_list, alpha_list = [], [], []
    input_view_depths_cuda = [torch.from_numpy(d).float().cuda() for d in input_view_depths]

    for idx, novel_cam in enumerate(novel_cams):
        pkg = render(novel_cam, gaussians, pipe, background)
        rgb, alpha, depth = pkg['render'], pkg['rend_alpha'], pkg['surf_depth']
        gs_depths.append(depth[0].detach())

        rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
        save_img_u8(rgb_np, os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(gs_output_dir, f'depth_frame{idx:06d}.tiff'))

        alpha_np = alpha[0].detach().cpu().numpy()
        np.save(os.path.join(gs_output_dir, f'alpha_{idx:06d}.npy'), alpha_np)
        alpha_vis_mask = (alpha_np > alpha_vis_thresh)
        alpha_list.append(alpha_vis_mask)
        save_img_u8(alpha_vis_mask, os.path.join(gs_output_dir, f'alpha_mask_frame{idx:06d}.png'))

        rgb_filtered = rgb_np * alpha_vis_mask[:, :, None]
        alpha_warp_u8 = (np.clip(rgb_filtered, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if _is_all_black_rgb_u8(alpha_warp_u8, thr=0):
            alpha_warp_u8 = _alpha_to_rgb_u8(alpha_np)
            print(f"[Info] Raw-GS View {idx:06d}: alpha_warp_frame is black -> replaced by alpha_frame visualization.")
        Image.fromarray(alpha_warp_u8).save(os.path.join(gs_output_dir, f'alpha_warp_frame{idx:06d}.png'))

        if only_warp_input_views:
            gs_points = depths_to_points_parallel(depth.unsqueeze(0), [novel_cam]).squeeze(0)
            visible_mask_tensor = get_visible_mask_for_input_views(input_viewpoints, input_view_depths_cuda, gs_points)
            visible_mask = visible_mask_tensor.reshape(rgb.shape[1], rgb.shape[2]).detach().cpu().numpy().astype(np.float32)
            save_img_u8(visible_mask, os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

            if np.sum(visible_mask) < (visible_mask.size * EMPTY_MASK_THRESHOLD):
                print(f"[Info] Raw-GS View {idx:06d}: visible_mask is nearly empty. Using ori_warp_frame.")
                rgb_filtered2 = rgb_np
            else:
                rgb_filtered2 = rgb_np * visible_mask[:, :, None]

            save_img_u8(rgb_filtered2, os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))

            none_visible_rate = 1 - visible_mask.sum() / (visible_mask.shape[0] * visible_mask.shape[1])
            none_visible_rate_list.append(none_visible_rate)

        print(f'Novel view {idx} save done!')
        _cuda_release(pkg, rgb, alpha, depth)

    if not only_warp_input_views:
        visibility_maps = visibility_grid.render_visibility_map(novel_cams, gs_depths)
        for idx in range(len(visibility_maps)):
            rgb_path = os.path.join(gs_output_dir, f'ori_warp_frame{idx:06d}.png')
            rgb_image = np.array(Image.open(rgb_path))

            vis_map = (visibility_maps[idx].detach().cpu().numpy() > 0.5)
            alpha_map = alpha_list[idx]
            vis_map = (vis_map & alpha_map).astype(np.float32)

            if np.sum(vis_map) < (vis_map.size * EMPTY_MASK_THRESHOLD):
                print(f"[Info] Raw-GS View {idx:06d}: vis_map is nearly empty. Using ori_warp_frame.")
                out_image = rgb_image
            else:
                out_image = rgb_image * vis_map[:, :, None]

            Image.fromarray(out_image.astype(np.uint8)).save(os.path.join(gs_output_dir, f'warp_frame{idx:06d}.png'))
            vis_u8 = (vis_map.astype(np.uint8) * 255)
            Image.fromarray(vis_u8).save(os.path.join(gs_output_dir, f'mask_frame{idx:06d}.png'))

            none_visible_rate = 1 - vis_map.sum() / (vis_map.shape[0] * vis_map.shape[1])
            none_visible_rate_list.append(none_visible_rate)

        print('Render visibility map done!')
        _cuda_release(visibility_maps)
        for d in gs_depths:
            _cuda_release(d)
        gs_depths = []

    # View Selection
    max_none_visible_thresh = 0.6
    if select_view_method == 'none_visible_rate':
        need_inpaint_views = [i for i in range(len(novel_cams)) if none_visible_rate_list[i] < max_none_visible_thresh]
    elif select_view_method == 'covisibility_rate':
        need_inpaint_views = select_need_inpaint_views(
            novel_cams,
            none_visible_rate_list,
            gaussians,
            int(args.select_inpaint_num),
            none_visible_rate_high_bound=max_none_visible_thresh,
            covisible_rate_high_bound=0.9
        )
    else:
        raise ValueError(f'Invalid select_view_method: {select_view_method}')

    print(f'Need inpaint views (base): {need_inpaint_views}')

    if args.delete_unselected_views:
        cleanup_unselected_raw_gs(gs_output_dir, total_views=len(novel_cams), keep_ids=need_inpaint_views)

    select_gs_output_dir = os.path.join(novel_views_save_root_path, 'select-gs')
    os.makedirs(select_gs_output_dir, exist_ok=True)
    need_inpaint_views_cams = [novel_cams[i] for i in need_inpaint_views]

    # Reload depths for base views
    need_inpaint_views_depths = []
    for ori_id in need_inpaint_views:
        d_path = os.path.join(gs_output_dir, f'depth_frame{ori_id:06d}.tiff')
        d = np.array(Image.open(d_path))
        need_inpaint_views_depths.append(torch.from_numpy(d).float().cuda())
    need_inpaint_views_depths = torch.stack(need_inpaint_views_depths, dim=0)

    # Curriculum
    stage_default_yaw = {1: 10.0, 2: 20.0, 3: 30.0}
    yaw_deg = args.curriculum_rot_deg if args.curriculum_rot_deg is not None else stage_default_yaw.get(args.see3d_stage, 10.0)

    curriculum_cache = []
    if float(yaw_deg) > 0:
        for bidx, base_cam in enumerate(need_inpaint_views_cams):
            ori_id = need_inpaint_views[bidx]

            base_alpha_mask_path = os.path.join(gs_output_dir, f'alpha_mask_frame{ori_id:06d}.png')
            if os.path.exists(base_alpha_mask_path):
                base_mask_u8 = _load_u8_mask(base_alpha_mask_path)
                base_top_missing = _top_missing_from_u8_mask(base_mask_u8, band_ratio=BASE_TOP_BAND_RATIO, on_thr=127)
            else:
                base_top_missing = 1.0

            if base_top_missing > BASE_TOP_MISSING_THRESH:
                print(f"[Curriculum][SkipBase] base#{bidx:03d} (ori_id={ori_id:06d}) "
                      f"top_missing={base_top_missing:.3f} > {BASE_TOP_MISSING_THRESH}")
                continue

            for name, cam in generate_curriculum_lr_cams(base_cam, yaw_deg=yaw_deg):
                pkg = render(cam, gaussians, pipe, background)

                rgb = pkg["render"]
                alpha = pkg.get("rend_alpha", None)
                depth_tensor = pkg["surf_depth"][0]

                rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
                rgb_u8 = (np.clip(rgb_np, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                depth_np = depth_tensor.detach().cpu().numpy().astype(np.float32)

                mask_u8 = ((depth_np > 0).astype(np.uint8) * 255)
                mask_black_ratio = _mask_black_ratio_u8(mask_u8, on_thr=127)
                if mask_black_ratio > CURR_MASK_BLACK_RATIO_THRESH:
                    print(f"[Curriculum][SkipView] base#{bidx:03d} {name}: "
                          f"mask_black_ratio={mask_black_ratio:.3f} > {CURR_MASK_BLACK_RATIO_THRESH}")
                    _cuda_release(pkg, rgb, depth_tensor, alpha)
                    continue

                if alpha is None:
                    alpha_map = (depth_tensor > 0).float().detach().cpu().numpy()
                else:
                    alpha_map = alpha[0].detach().cpu().numpy()

                alpha_mask_u8 = ((alpha_map > alpha_vis_thresh).astype(np.uint8) * 255)
                alpha_warp = (rgb_np * (alpha_mask_u8[..., None] / 255.0))
                alpha_warp_u8 = (np.clip(alpha_warp, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                if _is_all_black_rgb_u8(alpha_warp_u8, thr=0):
                    alpha_warp_u8 = _alpha_to_rgb_u8(alpha_map)
                    print(f"[Info] Curriculum candidate base#{bidx:03d} {name}: alpha_warp_frame black -> use alpha visualization.")

                curriculum_cache.append({
                    "bidx": bidx,
                    "name": name,
                    "cam": cam,
                    "rgb_u8": rgb_u8,
                    "depth_np": depth_np,
                    "mask_u8": mask_u8,
                    "alpha_map": alpha_map.astype(np.float32),
                    "alpha_mask_u8": alpha_mask_u8.astype(np.uint8),
                    "alpha_warp_u8": alpha_warp_u8.astype(np.uint8),
                })

                print(f"[Curriculum][Keep] base#{bidx:03d} (ori_id={ori_id:06d}, top_missing={base_top_missing:.3f}) "
                      f"{name}: mask_black_ratio={mask_black_ratio:.3f}")
                _cuda_release(pkg, rgb, depth_tensor, alpha)

    curriculum_cams = [x["cam"] for x in curriculum_cache]
    final_cams = need_inpaint_views_cams + curriculum_cams
    print(f"[Final] stage{args.see3d_stage}: base={len(need_inpaint_views_cams)}, curriculum={len(curriculum_cams)}, total={len(final_cams)}")

    # Point Cloud Generation
    all_depths_list = [need_inpaint_views_depths]
    if len(curriculum_cache) > 0:
        curr_depths = [torch.from_numpy(item["depth_np"]).float().cuda() for item in curriculum_cache]
        all_depths_list.append(torch.stack(curr_depths, dim=0))

    all_depths = torch.cat(all_depths_list, dim=0).unsqueeze(1)
    invalid_depth_mask = all_depths <= 1e-6
    all_depths[invalid_depth_mask] = 1e-3
    all_points = depths_to_points_parallel(all_depths, final_cams).reshape(-1, 3)
    all_points = all_points[~invalid_depth_mask.reshape(-1)]

    ply_path = os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_need_inpaint_views_points.ply')
    trimesh.PointCloud(all_points.detach().cpu().numpy()).export(ply_path)
    print(f'Saved all inpaint views points to {ply_path}')
    _cuda_release(all_depths, all_points, invalid_depth_mask)

    save_cameras = {'train_views': len(train_viewpoints)}
    for idx, cam in enumerate(need_inpaint_views_cams):
        ori_id = need_inpaint_views[idx]
        for key in ['R', 'T', 'FoVx', 'FoVy', 'image_width', 'image_height']:
            save_cameras[f'{key}_{idx:06d}'] = getattr(cam, key)

        for f in ['ori_warp_frame.png', 'depth_frame.tiff', 'alpha_.npy', 'alpha_mask_frame.png', 'alpha_warp_frame.png', 'warp_frame.png', 'mask_frame.png']:
            shutil.copy(
                os.path.join(gs_output_dir, f.replace('.', f'{ori_id:06d}.')),
                os.path.join(select_gs_output_dir, f.replace('.', f'{idx:06d}.'))
            )

    start_idx = len(need_inpaint_views_cams)
    if len(curriculum_cache) > 0:
        for j, item in enumerate(curriculum_cache):
            out_id = start_idx + j

            Image.fromarray(item["rgb_u8"]).save(os.path.join(select_gs_output_dir, f"warp_frame{out_id:06d}.png"))
            Image.fromarray(item["rgb_u8"]).save(os.path.join(select_gs_output_dir, f"ori_warp_frame{out_id:06d}.png"))
            Image.fromarray(item["mask_u8"]).save(os.path.join(select_gs_output_dir, f"mask_frame{out_id:06d}.png"))
            save_img_f32(item["depth_np"], os.path.join(select_gs_output_dir, f"depth_frame{out_id:06d}.tiff"))

            np.save(os.path.join(select_gs_output_dir, f"alpha_{out_id:06d}.npy"), item["alpha_map"].astype(np.float32))
            Image.fromarray(item["alpha_mask_u8"]).save(os.path.join(select_gs_output_dir, f"alpha_mask_frame{out_id:06d}.png"))
            Image.fromarray(item["alpha_warp_u8"]).save(os.path.join(select_gs_output_dir, f"alpha_warp_frame{out_id:06d}.png"))

            cam = item["cam"]
            for key in ['R', 'T', 'FoVx', 'FoVy', 'image_width', 'image_height']:
                save_cameras[f'{key}_{out_id:06d}'] = getattr(cam, key)

            print(f"Curriculum view {j} (ID {out_id}) saved. base#{item['bidx']:03d} {item['name']}")

    save_cameras['n_views'] = len(final_cams)
    np.savez(os.path.join(novel_views_save_root_path, f'stage{args.see3d_stage}_see3d_cameras.npz'), **save_cameras)

    print(f'See3D stage {args.see3d_stage} save done!')

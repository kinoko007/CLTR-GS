# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras

import gc
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from matcha.dm_scene.charts import (
    load_charts_data,
    schedule_regularization_factor_2,
    get_gaussian_parameters_from_pa_data,
    depths_to_points_parallel,
    depth2normal_parallel,
    normal2curv_parallel,
    voxel_downsample_gaussians,
)
from matcha.dm_regularization.depth import compute_depth_order_loss
from matcha.dm_utils.rendering import normal2curv

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import trimesh


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        torch.cuda.empty_cache()


def training(
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
    use_refined_charts, use_mip_filter, dense_data_path, use_chart_view_every_n_iter,
    normal_consistency_from, distortion_from,
    depthanythingv2_checkpoint_dir, depthanything_encoder,
    dense_regul, refine_depth_path, use_downsample_gaussians
):
    save_log_images = False
    save_log_images_every_n_iter = 200

    gaussian_points_count = []
    gaussian_points_iterations = []

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, shuffle=False)

    see3d_root_path = os.path.join(dataset.source_path, 'see3d_render')
    see3d_cam_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    if os.path.exists(see3d_cam_path):
        see3d_gs_cameras_list, _ = load_see3d_cameras(see3d_cam_path, inpaint_root_dir)
        print(f"[INFO] Loaded {len(see3d_gs_cameras_list)} See3D cameras from {see3d_cam_path}")
    else:
        see3d_gs_cameras_list = []
        print(f"[WARNING] No See3D cameras found at {see3d_cam_path}")

    use_chart_view_every_n_iter = 1
    print(f"[INFO] Charts will be used for regularization every {use_chart_view_every_n_iter} iteration(s).")

    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    if use_refined_charts:
        charts_data_path = f'{dataset.source_path}/refined_charts_data.npz'
    print("Using charts data from: ", charts_data_path)
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs']
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())

    print(f'[INFO]: Load plane-aware depth from: {refine_depth_path}')
    pa_depths = []
    pa_points_list = []
    pa_confident_maps_list = []

    input_view_num = len(scene.getTrainCameras())
    see3d_view_num = len(see3d_gs_cameras_list)
    training_view_num = input_view_num + see3d_view_num

    for idx in range(training_view_num):
        pa_depth_path = os.path.join(refine_depth_path, f'refine_depth_frame{idx:06d}.tiff')
        pa_point_path = os.path.join(refine_depth_path, f'refine_points_frame{idx:06d}.ply')
        pa_confident_map_path = os.path.join(refine_depth_path, f'confident_map_frame{idx:06d}.png')

        if not os.path.exists(pa_depth_path):
            raise FileNotFoundError(
                f"\n[FATAL] Plane-aware depth file missing for view {idx:06d}!\nPath: {pa_depth_path}\n"
                f"Please generate the missing depth file first (run plane-refine depth script)."
            )
        if not os.path.exists(pa_point_path):
            raise FileNotFoundError(
                f"\n[FATAL] Point cloud file missing for view {idx:06d}!\nPath: {pa_point_path}\n"
                f"Please generate the missing point cloud first."
            )
        if not os.path.exists(pa_confident_map_path):
            raise FileNotFoundError(
                f"\n[FATAL] Confidence map file missing for view {idx:06d}!\nPath: {pa_confident_map_path}\n"
                f"Please generate the missing confidence map first."
            )

        pa_depth = Image.open(pa_depth_path)
        pa_depth = np.array(pa_depth)
        pa_depth = torch.from_numpy(pa_depth).cuda()
        pa_depths.append(pa_depth)

        pa_point = trimesh.load(pa_point_path)
        pa_point = np.array(pa_point.vertices)
        pa_points_list.append(pa_point)

        pa_confident_map = Image.open(pa_confident_map_path)
        pa_confident_map = np.array(pa_confident_map) / 255
        pa_confident_map = torch.from_numpy(pa_confident_map).cuda()
        pa_confident_maps_list.append(pa_confident_map)

    max_gaussians_num = 10_000_000
    print(f"Max gaussians num: {max_gaussians_num}, use downsample gaussians: {use_downsample_gaussians}")

    input_view_depths = pa_depths[:input_view_num]
    input_view_depths_stack = torch.stack(input_view_depths, dim=0).cuda()
    _images = [cam.original_image.cuda().permute(1, 2, 0) for cam in scene.getTrainCameras()]
    pa_points = depths_to_points_parallel(input_view_depths_stack, scene.getTrainCameras())
    N, H, W = input_view_depths_stack.shape
    pa_points = pa_points.reshape(N, H, W, 3)

    max_init_gs_input_view_num = 50
    if input_view_num > max_init_gs_input_view_num:
        print(f'[INFO]: Input view num too large: {input_view_num}, use {max_init_gs_input_view_num} views for init')
        init_view_ids = np.linspace(0, input_view_num - 1, max_init_gs_input_view_num, dtype=int)
        init_pa_points = [pa_points[i] for i in init_view_ids]
        init_pa_points_stack = torch.stack(init_pa_points, dim=0).cuda()
        init_images = [_images[i] for i in init_view_ids]
    else:
        init_pa_points_stack = pa_points
        init_images = _images

    input_view_gaussian_params = get_gaussian_parameters_from_pa_data(
        pa_points=init_pa_points_stack,
        images=init_images,
        conf_th=-1.,
        ratio_th=5.,
        normal_scale=1e-10,
        normalized_scales=0.5,
    )

    if see3d_view_num > 0:
        see3d_view_depths = pa_depths[input_view_num:]
        see3d_view_depths_stack = torch.stack(see3d_view_depths, dim=0).cuda()
        _see3d_images = [cam.original_image.cuda().permute(1, 2, 0) for cam in see3d_gs_cameras_list]

        see3d_points = depths_to_points_parallel(see3d_view_depths_stack, see3d_gs_cameras_list)
        N2, H2, W2 = see3d_view_depths_stack.shape
        see3d_points = see3d_points.reshape(N2, H2, W2, 3)

        see3d_gaussian_params = get_gaussian_parameters_from_pa_data(
            pa_points=see3d_points,
            images=_see3d_images,
            conf_th=-1.,
            ratio_th=5.,
            normal_scale=1e-10,
            normalized_scales=0.5,
        )

        gaussian_params = {}
        for key in input_view_gaussian_params.keys():
            gaussian_params[key] = torch.cat([input_view_gaussian_params[key], see3d_gaussian_params[key]], dim=0)
    else:
        gaussian_params = input_view_gaussian_params

    if len(gaussian_params['means']) > max_gaussians_num and use_downsample_gaussians:
        sample_idx, downsample_factor = voxel_downsample_gaussians(gaussian_params, voxel_size=0.002)
        print(f"Downsampled {len(gaussian_params['means'])} -> {len(sample_idx)}")
    else:
        sample_idx = torch.arange(len(gaussian_params['means']))
        downsample_factor = 1.0
        print(f"Not downsampling, using all {len(gaussian_params['means'])}")

    print(f"Final number of gaussians: {len(sample_idx)}")
    _means = gaussian_params['means'][sample_idx]
    _scales = gaussian_params['scales'][..., :2][sample_idx] * downsample_factor
    _quaternions = gaussian_params['quaternions'][sample_idx]
    _colors = gaussian_params['colors'][sample_idx]
    gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale)
    print("[INFO] Gaussians created.")

    del _means, _scales, _quaternions, _colors, gaussian_params, sample_idx
    gc.collect()
    torch.cuda.empty_cache()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0

    input_cams = scene.getTrainCameras()

    input_refine_depths = torch.stack(pa_depths[:input_view_num], dim=0).cuda()
    input_pseudo_confs = torch.stack(pa_confident_maps_list[:input_view_num], dim=0).cuda()
    input_world_view_transforms = torch.stack([c.world_view_transform for c in input_cams])
    input_full_proj_transforms = torch.stack([c.full_proj_transform for c in input_cams])
    input_prior_normals = depth2normal_parallel(
        input_refine_depths,
        world_view_transforms=input_world_view_transforms,
        full_proj_transforms=input_full_proj_transforms
    ).permute(0, 3, 1, 2)
    input_prior_curvs = normal2curv_parallel(input_prior_normals, torch.ones_like(input_prior_normals[:, 0:1]))
    print('Input pointmap loaded!')

    if see3d_view_num > 0:
        see3d_refine_depths = torch.stack(pa_depths[input_view_num:], dim=0).cuda()
        see3d_pseudo_confs = torch.stack(pa_confident_maps_list[input_view_num:], dim=0).cuda()
        see3d_world_view_transforms = torch.stack([c.world_view_transform for c in see3d_gs_cameras_list])
        see3d_full_proj_transforms = torch.stack([c.full_proj_transform for c in see3d_gs_cameras_list])
        see3d_prior_normals = depth2normal_parallel(
            see3d_refine_depths,
            world_view_transforms=see3d_world_view_transforms,
            full_proj_transforms=see3d_full_proj_transforms
        ).permute(0, 3, 1, 2)
        see3d_prior_curvs = normal2curv_parallel(see3d_prior_normals, torch.ones_like(see3d_prior_normals[:, 0:1]))
        print('See3D pointmap loaded!')

        total_views_list = input_cams + see3d_gs_cameras_list
        total_confs_list = [input_pseudo_confs[i] for i in range(len(input_pseudo_confs))] + \
                           [see3d_pseudo_confs[i].unsqueeze(0) for i in range(len(see3d_pseudo_confs))]
        total_depths_list = [input_refine_depths[i] for i in range(len(input_refine_depths))] + \
                            [see3d_refine_depths[i].unsqueeze(0) for i in range(len(see3d_refine_depths))]
        total_normals_list = [input_prior_normals[i] for i in range(len(input_prior_normals))] + \
                             [see3d_prior_normals[i] for i in range(len(see3d_prior_normals))]
        total_curvs_list = [input_prior_curvs[i] for i in range(len(input_prior_curvs))] + \
                           [see3d_prior_curvs[i] for i in range(len(see3d_prior_curvs))]
    else:
        total_views_list = input_cams
        total_confs_list = [input_pseudo_confs[i] for i in range(len(input_pseudo_confs))]
        total_depths_list = [input_refine_depths[i] for i in range(len(input_refine_depths))]
        total_normals_list = [input_prior_normals[i] for i in range(len(input_prior_normals))]
        total_curvs_list = [input_prior_curvs[i] for i in range(len(input_prior_curvs))]

    print(f"[INFO] Total views: {len(total_views_list)}, input: {len(input_cams)}, see3d: {len(see3d_gs_cameras_list)}")

    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.set_mip_filter(use_mip_filter)
        gaussians.compute_mip_filter(cameras=total_views_list)

    print(f"[INFO] Normal consistency from {normal_consistency_from} lambda_normal {opt.lambda_normal}")
    print(f"[INFO] Distortion from {distortion_from} lambda_dist {opt.lambda_dist}")

    use_depth_order_regularization = True

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = total_views_list.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if viewpoint_cam in see3d_gs_cameras_list:
            loss_img = loss_img * 0.01 #0.01->0.1/0.001

        lambda_normal = opt.lambda_normal if iteration > normal_consistency_from else 0.0
        lambda_dist = opt.lambda_dist if iteration > distortion_from else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()

        total_loss = loss_img + dist_loss + normal_loss

        vp_index = total_views_list.index(viewpoint_cam)
        current_depth = total_depths_list[vp_index]
        current_normal = total_normals_list[vp_index]
        current_curv = total_curvs_list[vp_index]

        surf_depth = render_pkg['surf_depth']
        rend_curvature = normal2curv(render_pkg['rend_normal'], torch.ones_like(render_pkg['rend_normal'][0:1]))

        initial_regularization_factor = 0.5
        regularization_factor = schedule_regularization_factor_2(iteration, initial_regularization_factor)
        lambda_prior_depth = regularization_factor * 0.75
        lambda_prior_depth_derivative = regularization_factor * 0.5
        lambda_prior_normal = regularization_factor * 0.5
        lambda_prior_curvature = regularization_factor * 0.25

        depth_prior_loss = lambda_prior_depth * (torch.log(1. + (current_depth - surf_depth).abs())).mean()
        if lambda_prior_depth_derivative > 0:
            depth_prior_loss += (lambda_prior_depth_derivative * (1. - (surf_normal * current_normal).sum(dim=0))).mean()

        normal_prior_loss = lambda_prior_normal * (1. - (rend_normal * current_normal).sum(dim=0)).mean()
        curv_prior_loss = lambda_prior_curvature * (current_curv - rend_curvature).abs().mean()

        if use_depth_order_regularization:
            lambda_depth_order = 0.
            if iteration > 1500:
                lambda_depth_order = 1.
            if iteration > 3000:
                lambda_depth_order = 0.1
            if iteration > 4500:
                lambda_depth_order = 0.01
            if iteration > 6000:
                lambda_depth_order = 0.001

            if lambda_depth_order > 0:
                depth_order_prior_loss = lambda_depth_order * compute_depth_order_loss(
                    depth=surf_depth,
                    prior_depth=current_depth.to(surf_depth.device),
                    scene_extent=gaussians.spatial_lr_scale,
                    max_pixel_shift_ratio=0.05,
                    normalize_loss=True,
                    log_space=True,
                    log_scale=20.,
                    reduction="mean",
                    debug=False,
                )
            else:
                depth_order_prior_loss = torch.zeros_like(loss_img.detach())
            depth_prior_loss = depth_prior_loss + depth_order_prior_loss

        total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss

        lambda_anisotropy = 0.1
        anisotropy_max_ratio = 5.0
        if lambda_anisotropy > 0.:
            gs_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gs_scaling.max(dim=1).values / gs_scaling.min(dim=1).values, anisotropy_max_ratio)
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss

        total_loss = total_loss + total_regularization_loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss_img.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log
            if lambda_anisotropy > 0.:
                ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log

            if iteration % 10 == 0:
                current_points = len(gaussians.get_xyz.detach())
                gaussian_points_count.append(current_points)
                gaussian_points_iterations.append(iteration)

                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "dist": f"{ema_dist_for_log:.5f}",
                    "norm": f"{ema_normal_for_log:.5f}",
                    "Pts": f"{current_points}",
                    "p_d": f"{ema_prior_depth_for_log:.5f}",
                    "p_n": f"{ema_prior_normal_for_log:.5f}",
                    "p_c": f"{ema_prior_curvature_for_log:.5f}",
                    "aniso": f"{ema_prior_anisotropy_for_log:.5f}",
                })
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                                               size_threshold, iteration, opt.grace_period)
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=total_views_list)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                ######
                if opt.use_lbo_planning and iteration >= opt.lbo_planning_start_iter and iteration % opt.lbo_planning_interval == 0:
                    gaussians.lbo_gaussian_planning(scene, pipe, background, opt, iteration)
                ######
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte()
                                                     .permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {"#": gaussians.get_opacity.shape[0], "loss": ema_loss_for_log}
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

    if len(gaussian_points_count) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(gaussian_points_iterations, gaussian_points_count)
        plt.xlabel('Iterations')
        plt.ylabel('Number of Gaussian Points')
        plt.title('Gaussian Points Count During Training')
        plt.grid(True)
        plt.savefig(f"{dataset.model_path}/gaussian_points_count.png")
        plt.close()

    print("Training complete.")


if __name__ == "__main__":
    import time
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument(
        "--mast3r_scene",
        dest="source_path",
        type=str,
        default=None,
        help="Alias for --source_path (MASt3R scene directory)."
    )
    parser.add_argument(
        "--output_path", "-o",
        dest="model_path",
        type=str,
        default=None,
        help="Alias for --model_path (output directory)."
    )

    parser.add_argument("--refine_depth_path", type=str, required=True)
    parser.add_argument("--use_downsample_gaussians", action="store_true", help="Use downsample gaussians")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--use_refined_charts", action="store_true", default=False)
    parser.add_argument("--use_mip_filter", action="store_true", default=False)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument("--use_chart_view_every_n_iter", type=int, default=999_999)

    parser.add_argument("--normal_consistency_from", type=int, default=3500)
    parser.add_argument("--distortion_from", type=int, default=1500)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='../Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default',
                        help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not getattr(args, "source_path", None):
        raise ValueError("Missing scene path. Please provide --mast3r_scene or --source_path.")
    if not getattr(args, "model_path", None):
        raise ValueError("Missing output path. Please provide --output_path/-o or --model_path.")

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if args.port is None:
        current_time = time.strftime("%H%M%S", time.localtime())[2:]
        args.port = int(current_time)
        print(f"Randomly selected port: {args.port}")

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.use_refined_charts, args.use_mip_filter,
        args.dense_data_path, args.use_chart_view_every_n_iter,
        args.normal_consistency_from, args.distortion_from,
        args.depthanythingv2_checkpoint_dir, args.depthanything_encoder,
        args.dense_regul, args.refine_depth_path, args.use_downsample_gaussians,
    )

    print("\nTraining complete.")

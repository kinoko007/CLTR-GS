# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import kornia
import trimesh

from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement

def _gu():
    import utils.general_utils as gu
    return gu

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            gu = _gu()
            scaling_3d = torch.cat([
                scaling * scaling_modifier,
                torch.ones((scaling.shape[0], 1), device=scaling.device, dtype=scaling.dtype)
            ], dim=-1)
            RS = gu.build_scaling_rotation(scaling_3d, rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: _gu().inverse_sigmoid(x)

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)   # 2DGS: (N,2)
        self._rotation = torch.empty(0)  # quat
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self._creation_iter = torch.empty(0, dtype=torch.long, device="cuda")
        self._is_protected = torch.empty(0, dtype=torch.bool, device="cuda")
        self._denom_prune_min = 0

        self.use_mip_filter = False
        self.mip_filter = torch.empty(0)
        self.origin_xyz = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity, self.max_radii2D,
            self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(), self.spatial_lr_scale
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
         self._scaling, self._rotation, self._opacity, self.max_radii2D,
         xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def freeze_params(self):
        original_point_count = self._xyz.shape[0]
        print(f"Freezing {original_point_count} points in GaussianModel")
        
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
        if self.optimizer is not None:
            updated_param_groups = []
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    group["params"] = [self._xyz]
                elif group["name"] == "f_dc":
                    group["params"] = [self._features_dc]
                elif group["name"] == "f_rest":
                    group["params"] = [self._features_rest]
                elif group["name"] == "opacity":
                    group["params"] = [self._opacity]
                elif group["name"] == "scaling":
                    group["params"] = [self._scaling]
                elif group["name"] == "rotation":
                    group["params"] = [self._rotation]
                group["lr"] = 0.0
                updated_param_groups.append(group)
            
            self.optimizer = torch.optim.Adam(
                updated_param_groups,
                lr=0.0,
                eps=1e-15
            )

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling)
        if self.use_mip_filter:
            scales = torch.square(scales) + torch.square(self.mip_filter)
            scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity)
        if self.use_mip_filter:
            scales = self.scaling_activation(self._scaling)
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.mip_filter)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacity = opacity * coef[..., None]
        return opacity

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)   # 2DGS: (sx, sy)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.origin_xyz = fused_point_cloud.clone()
        num_pts = self.get_xyz.shape[0]
        self._creation_iter = torch.zeros(num_pts, dtype=torch.long, device="cuda")
        self._is_protected = torch.zeros(num_pts, dtype=torch.bool, device="cuda")

    def create_from_parameters(self, _means, _scales, _quaternions, _colors, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = _means
        fused_color = RGB2SH(_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.log(_scales)
        rots = _quaternions
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.origin_xyz = fused_point_cloud.clone()
        num_pts = self.get_xyz.shape[0]
        self._creation_iter = torch.zeros(num_pts, dtype=torch.long, device="cuda")
        self._is_protected = torch.zeros(num_pts, dtype=torch.bool, device="cuda")

    def set_mip_filter(self, use_mip_filter: bool):
        self.use_mip_filter = use_mip_filter
        if use_mip_filter and self.mip_filter.numel() == 0:
            self.mip_filter = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    @torch.no_grad()
    def compute_mip_filter(self, cameras, znear=0.2, filter_variance=0.2):
        if not self.use_mip_filter:
            print("[WARNING] Computing mip filter but mip filter is currently disabled.")
            return

        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        focal_length = 0.

        for camera in cameras:
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            xyz_cam = xyz @ R + T[None, :]
            valid_depth = xyz_cam[:, 2] > znear

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0

            in_screen = torch.logical_and(
                torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15),
                torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height)
            )
            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x

        distance[~valid_points] = distance[valid_points].max()
        self.mip_filter = (distance / focal_length * (filter_variance ** 0.5))[..., None]

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        if self.use_mip_filter:
            mip_filter = self.mip_filter.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        if self.use_mip_filter:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, mip_filter), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if "mip_filter" in [p.name for p in plydata.elements[0].properties]:
            mip_filter = np.asarray(plydata.elements[0]["mip_filter"])[..., np.newaxis]
            self.set_mip_filter(True)
            self.mip_filter = torch.tensor(mip_filter, dtype=torch.float, device="cuda")
            print("[INFO] Loading mip filter from ply file.")
        else:
            print("[INFO] No mip filter found in ply file.")
            self.use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        self.origin_xyz = self._xyz.detach().clone()
        num_pts = self.get_xyz.shape[0]
        self._creation_iter = torch.zeros(num_pts, dtype=torch.long, device="cuda")
        self._is_protected = torch.zeros(num_pts, dtype=torch.bool, device="cuda")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('mip_filter')
        return l

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iter=0):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.origin_xyz = self.origin_xyz[valid_points_mask]

        if self._creation_iter.numel() == mask.numel():
            self._creation_iter = self._creation_iter[valid_points_mask]
            self._is_protected = self._is_protected[valid_points_mask]

        if self.use_mip_filter and self.mip_filter.numel() == mask.numel():
            self.mip_filter = self.mip_filter[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            ext = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(ext)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(ext)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], ext), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], ext), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_origin_xyz, creation_iter):
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest,
             "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.origin_xyz = torch.cat((self.origin_xyz, new_origin_xyz), dim=0)

        num_new = new_xyz.shape[0]
        new_creation_iters = torch.full((num_new,), creation_iter, dtype=torch.long, device="cuda")
        new_protected_status = torch.ones(num_new, dtype=torch.bool, device="cuda")
        self._creation_iter = torch.cat((self._creation_iter, new_creation_iters), dim=0)
        self._is_protected = torch.cat((self._is_protected, new_protected_status), dim=0)

        if self.use_mip_filter:
            new_mip = torch.zeros((num_new, 1), device="cuda")
            self.mip_filter = torch.cat((self.mip_filter, new_mip), dim=0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._denom_prune_min = training_args.denom_prune_min if hasattr(training_args, "denom_prune_min") else 5

        gu = _gu()
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = gu.get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for g in self.optimizer.param_groups:
            if g["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                g['lr'] = lr
                return lr

    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        gu = _gu()
        n_init = self.get_xyz.shape[0]
        padded = torch.zeros((n_init), device="cuda")
        padded[:grads.shape[0]] = grads.squeeze()
        selected = torch.where(padded >= grad_threshold, True, False)
        selected = torch.logical_and(selected, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones((stds.shape[0], 1), device=stds.device)], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = gu.build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_fdc = self._features_dc[selected].repeat(N, 1, 1)
        new_frt = self._features_rest[selected].repeat(N, 1, 1)
        new_opa = self._opacity[selected].repeat(N, 1)
        new_org = self.origin_xyz[selected].repeat(N, 1)

        self.densification_postfix(new_xyz, new_fdc, new_frt, new_opa, new_scaling, new_rotation, new_org, iter)

        prune_filter = torch.cat((selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iter):
        selected = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected = torch.logical_and(selected, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected]
        new_fdc = self._features_dc[selected]
        new_frt = self._features_rest[selected]
        new_opa = self._opacity[selected]
        new_scal = self._scaling[selected]
        new_rot = self._rotation[selected]
        new_org = self.origin_xyz[selected]

        self.densification_postfix(new_xyz, new_fdc, new_frt, new_opa, new_scal, new_rot, new_org, iter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter, grace_period):
        use_mip_filter = self.use_mip_filter
        if use_mip_filter:
            self.set_mip_filter(False)

        gu = _gu()
        if self._creation_iter.numel() > 0:
            elapsed = iter - self._creation_iter
            self._is_protected[elapsed > grace_period] = False

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, iter)
        self.densify_and_split(grads, max_grad, extent, iter)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = torch.logical_and(prune_mask, self.denom.squeeze() > self._denom_prune_min)
        if self._is_protected.numel() == prune_mask.numel():
            prune_mask = torch.logical_and(prune_mask, ~self._is_protected)
        '''
        try:
            nbr = self.inplane_knn(k=8, candidate_k=64)
            neigh_pts = self._xyz[nbr]
            normals = gu.build_rotation(self._rotation)[..., 2]
            neigh_n = normals[nbr]
            cosang = (neigh_n * normals.unsqueeze(1)).sum(-1).mean(-1)
            centered = neigh_pts - neigh_pts.mean(dim=1, keepdim=True)
            C = torch.bmm(centered.transpose(1, 2), centered) / (centered.shape[1] - 1)
            evals = torch.linalg.eigvalsh(C)
            lam_min = evals[:, 0]
            thin_mask = (cosang > 0.98) & (lam_min < torch.quantile(lam_min, 0.2))
            prune_mask = torch.logical_and(prune_mask, ~thin_mask)
        except Exception as e:
            print(f"[WARNING] Geometric thin point filter failed: {e}")
            pass

        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_vs), big_ws)
        '''
        self.prune_points(prune_mask, iter)
        torch.cuda.empty_cache()

        if use_mip_filter:
            self.set_mip_filter(True)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        op_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(op_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad()
    def get_tetra_points(
        self,
        downsample_ratio: float = None,
        gaussian_flatness: float = 1e-3,
        return_idx: bool = False,
        points_idx: torch.Tensor = None,
    ):
        M = trimesh.creation.box()
        M.vertices *= 2

        rots = _gu().build_rotation(self._rotation)
        scales_3d = torch.nn.functional.pad(
            self.get_scaling,
            (0, 1),
            mode="constant",
            value=gaussian_flatness,
        )
        print(f"[INFO] Padding 2D scaling with {gaussian_flatness} for tetra points: {scales_3d[0]}")

        if (downsample_ratio is None) and (points_idx is None):
            xyz = self.get_xyz
            scale = scales_3d * 3.
        else:
            if points_idx is None:
                print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                xyz_idx = torch.randperm(self.get_xyz.shape[0])[:int(self.get_xyz.shape[0] * downsample_ratio)]
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
            else:
                downsample_ratio = len(points_idx) / len(self.get_xyz)
                xyz_idx = points_idx
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")

        vertices = torch.from_numpy(M.vertices.T).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices)
        vertices = vertices + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        vertices = torch.cat([vertices, xyz], dim=0)

        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)

        if return_idx:
            if downsample_ratio is None:
                print("[WARNING] return_idx might not be needed when downsample_ratio is None")
                xyz_idx = torch.arange(self.get_xyz.shape[0])
            return vertices, vertices_scale, xyz_idx
        else:
            return vertices, vertices_scale

    def gs_scale_loss(self, max_scale_thresh=0.05):
        max_scale = self.get_scaling.max(dim=1).values
        excess = torch.clamp(max_scale - max_scale_thresh, min=0.0)
        loss = torch.sum(excess ** 2)
        return loss

    def _quat_slerp(self, q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor):
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
        mask = (dot < 0.0)
        q2 = torch.where(mask, -q2, q2)
        dot = torch.where(mask, -dot, dot)

        DOT_THRESHOLD = 0.9995
        t = t.unsqueeze(-1)

        linear = (dot > DOT_THRESHOLD).squeeze(-1)
        out_lin = F.normalize(q1 + t * (q2 - q1), dim=-1)

        theta_0 = torch.acos(dot.clamp(-1.0, 1.0))
        sin_0 = torch.sin(theta_0)
        theta = theta_0 * t
        s0 = torch.sin(theta_0 - theta) / (sin_0 + 1e-9)
        s1 = torch.sin(theta) / (sin_0 + 1e-9)
        out_slerp = s0 * q1 + s1 * q2

        return torch.where(linear.unsqueeze(-1), out_lin, out_slerp)

    def _inplane_metric(self):
        gu = _gu()
        R = gu.build_rotation(self._rotation)
        U = R[:, :, :2]
        S = self.scaling_activation(self._scaling)
        inv2 = torch.diag_embed(1.0 / (S * S + 1e-9))
        return U @ inv2 @ U.transpose(1, 2)

    @torch.no_grad()
    def inplane_knn(
        self,
        k: int = 16,
        candidate_k: int = 128,
        subsample_indices: torch.Tensor = None,
        refine_on_gpu: bool = True,
        return_global: bool = True,
    ):
        if subsample_indices is None:
            subsample_indices = torch.arange(self.get_xyz.shape[0], device="cuda", dtype=torch.long)
        else:
            subsample_indices = subsample_indices.to("cuda", dtype=torch.long)

        M = subsample_indices.numel()
        if M == 0 or k <= 0:
            return torch.empty((0, k), dtype=torch.long, device="cuda")

        P_all = self.get_xyz.detach()
        P_sub = P_all[subsample_indices]

        C = min(max(candidate_k, k + 1), M)
        P_cpu = P_sub.cpu().numpy().astype(np.float32, copy=False)
        tree = cKDTree(P_cpu)
        _, cand_idx_np = tree.query(P_cpu, k=C, workers=-1)
        if C == 1:
            cand_idx_np = cand_idx_np[:, None]
        cand_idx_np = cand_idx_np[:, 1:] if cand_idx_np.shape[1] > 1 else cand_idx_np
        while cand_idx_np.shape[1] < k:
            need = k - cand_idx_np.shape[1]
            rep = cand_idx_np[:, :min(cand_idx_np.shape[1], need)]
            cand_idx_np = np.concatenate([cand_idx_np, rep], axis=1)

        cand_idx = torch.from_numpy(cand_idx_np[:, :max(k, 1)]).to("cuda", dtype=torch.long)

        if refine_on_gpu:
            M_metric = self._inplane_metric()[subsample_indices]
            cand_pts = P_sub[cand_idx]
            diff = cand_pts - P_sub[:, None, :]
            tmp = torch.matmul(diff.unsqueeze(2), M_metric.unsqueeze(1))
            d2 = torch.matmul(tmp, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            topk_val, topk_loc = torch.topk(d2, k=k, largest=False, dim=1)
            nn_loc = torch.gather(cand_idx, 1, topk_loc)
        else:
            nn_loc = cand_idx[:, :k].contiguous()

        if return_global:
            nn_global = subsample_indices[nn_loc]
            return nn_global
        else:
            return nn_loc

    def find_significant_components(self, min_component_size, k, neighbor_indices=None):
        num_points = self.get_xyz.shape[0]
        if num_points == 0:
            return torch.zeros(0, dtype=torch.bool, device="cuda")
        if neighbor_indices is None or neighbor_indices.shape[0] != num_points or neighbor_indices.shape[1] < k:
            if num_points <= k:
                return torch.zeros(num_points, dtype=torch.bool, device="cuda")
            neighbor_indices = self.inplane_knn(k=k, candidate_k=256)
        elif neighbor_indices.shape[1] > k:
            neighbor_indices = neighbor_indices[:, :k]
        if neighbor_indices.numel() == 0:
            return torch.ones(num_points, dtype=torch.bool, device="cuda") if min_component_size > 1 else torch.zeros(num_points, dtype=torch.bool, device="cuda")

        source_indices = torch.arange(num_points, device="cuda").unsqueeze(1).expand_as(neighbor_indices)
        u, v = source_indices.flatten(), neighbor_indices.flatten()
        valid = (u != v)
        u, v = u[valid], v[valid]

        edges = torch.stack([u, v], dim=1).cpu().numpy()
        if edges.shape[0] == 0:
            return torch.ones(num_points, dtype=torch.bool, device="cuda") if min_component_size > 1 else torch.zeros(num_points, dtype=torch.bool, device="cuda")
        data = np.ones(edges.shape[0], dtype=bool)
        adj = csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(num_points, num_points))
        n_comp, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        if n_comp == 0:
            return torch.zeros(num_points, dtype=torch.bool, device="cuda")
        labels_t = torch.from_numpy(labels).long().to("cuda", non_blocking=True)
        sizes = torch.bincount(labels_t)
        sig_ids = torch.where(sizes >= min_component_size)[0]
        is_sig = torch.zeros(n_comp, dtype=torch.bool, device="cuda")
        is_sig[sig_ids] = True
        keep = is_sig[labels_t]
        prune_mask = ~keep
        return prune_mask

    def analyze_geometric_components(self, min_component_size, k=8,
                                     max_points=120_000, candidate_k=64, refine_on_gpu=False):
        N = self.get_xyz.shape[0]
        if N == 0:
            return None, None, None

        sub_idx = self._select_planning_subset(max_points)
        M = sub_idx.shape[0]
        if M < max(3, k + 1):
            return None, None, None

        nn_idx_local = self.inplane_knn(k=k, candidate_k=candidate_k,
                                        subsample_indices=sub_idx,
                                        refine_on_gpu=refine_on_gpu,
                                        return_global=False)

        src = torch.arange(M, device="cuda", dtype=torch.long).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = nn_idx_local.reshape(-1)
        mask = src != dst
        src = src[mask].cpu().numpy()
        dst = dst[mask].cpu().numpy()

        data = np.ones_like(src, dtype=np.uint8)
        adj = csr_matrix((data, (src, dst)), shape=(M, M))
        n_comp, labels_sub = connected_components(csgraph=adj, directed=False, return_labels=True)
        labels_sub = torch.from_numpy(labels_sub).to("cuda", dtype=torch.long)

        P_full = self.get_xyz.detach().cpu().numpy().astype(np.float32)
        P_sub = self.get_xyz.detach()[sub_idx].cpu().numpy().astype(np.float32)
        tree_sub = cKDTree(P_sub)
        _, nn_sub_of_full = tree_sub.query(P_full, k=1, workers=-1)
        nn_sub_of_full = torch.from_numpy(nn_sub_of_full).to("cuda", dtype=torch.long)

        labels_full = labels_sub[nn_sub_of_full]
        comp_sizes_full = torch.bincount(labels_full, minlength=n_comp)
        significant_mask = comp_sizes_full >= min_component_size
        return labels_full, comp_sizes_full, significant_mask

    def _select_planning_subset(self, max_points: int):
        N = self.get_xyz.shape[0]
        if N <= max_points:
            return torch.arange(N, device="cuda", dtype=torch.long)
        perm = torch.randperm(N, device="cuda")
        return perm[:max_points]

    def _bridge_on_plane(self, idx_a, idx_b, m):
        gu = _gu()
        Ra = gu.build_rotation(self._rotation[idx_a.view(-1)])
        Rb = gu.build_rotation(self._rotation[idx_b.view(-1)])
        n = F.normalize((Ra.mean(dim=0)[:, 2] + Rb.mean(dim=0)[:, 2]), p=2, dim=0)

        u = torch.cross(n, torch.tensor([0., 1., 0.], device="cuda", dtype=torch.float32))
        if torch.linalg.norm(u) < 1e-6:
            u = torch.tensor([1., 0., 0.], device="cuda", dtype=torch.float32)
        u = F.normalize(u, p=2, dim=0)
        v = F.normalize(torch.cross(n, u), p=2, dim=0)
        B = torch.stack([u, v, n], dim=1)

        def _proj(x):
            q = B.t() @ x.T
            q[2] = 0
            return (B @ q).T

        pa2 = _proj(self._xyz[idx_a])
        pb2 = _proj(self._xyz[idx_b])
        al = torch.linspace(0, 1, m + 2, device="cuda")[1:-1].unsqueeze(1)
        return pa2 * (1 - al) + pb2 * al

    def _create_bridging_gaussians(self, points1, points2, count_fallback):
        if points1.shape[0] == 0 or points2.shape[0] == 0:
            return {}

        d = torch.cdist(points1, points2)
        argmin = d.argmin()
        i = (argmin // d.shape[1]).item()
        j = (argmin % d.shape[1]).item()
        p1 = points1[i]
        p2 = points2[j]

        S = self.scaling_activation(self._scaling)
        s_med = torch.median(0.5 * (S[:, 0] + S[:, 1])).item()
        gap = torch.linalg.norm(p1 - p2).item()
        m = int(max(6, min(12, gap / max(s_med, 1e-3))))
        if m <= 0:
            m = count_fallback

        all_pts = self._xyz.detach()
        _, idx_p1 = cKDTree(all_pts.cpu().numpy()).query(p1.detach().cpu().numpy(), k=1)
        _, idx_p2 = cKDTree(all_pts.cpu().numpy()).query(p2.detach().cpu().numpy(), k=1)
        idx_p1 = torch.tensor(idx_p1, device="cuda", dtype=torch.long)
        idx_p2 = torch.tensor(idx_p2, device="cuda", dtype=torch.long)

        brid_xyz = self._bridge_on_plane(idx_p1, idx_p2, m)
        qa = self._rotation[idx_p1].unsqueeze(0).repeat(m, 1)
        qb = self._rotation[idx_p2].unsqueeze(0).repeat(m, 1)
        ts = torch.linspace(0, 1, m, device="cuda", dtype=brid_xyz.dtype)
        brid_rot = self._quat_slerp(qa, qb, ts)

        return {'xyz': brid_xyz, 'rotation': brid_rot}

    def _create_filling_gaussians(self, boundary_indices, k):
        if boundary_indices.numel() == 0:
            return {}
        nbr = self.inplane_knn(k=k, candidate_k=256, subsample_indices=boundary_indices,
                               return_global=False)
        if nbr.numel() == 0:
            return {}
        neigh_pts = self._xyz[boundary_indices[nbr]]
        new_xyz = neigh_pts.mean(dim=1)
        return {'xyz': new_xyz}

    def lbo_gaussian_planning(self, scene, pipe, background, opt, iteration):
        from gaussian_renderer import render
        print(f"\n[ITER {iteration}] Running 2DGS-Topo Planning...")

        comp_labels, comp_sizes, sig_mask = self.analyze_geometric_components(
            min_component_size=opt.lbo_planning_min_component_size,
            k=getattr(opt, "knn_k", 16),
            max_points=getattr(opt, "topo_max_points", 120000),
            candidate_k=getattr(opt, "topo_candidate_k", 64),
            refine_on_gpu=getattr(opt, "knn_refine_on_gpu", False)
        )
        if comp_labels is None:
            print("[INFO] Topo analysis skip.")
            return
        prune_mask = torch.zeros_like(comp_labels, dtype=torch.bool, device="cuda")
        num_to_prune = prune_mask.sum().item()

        train_cams = scene.getTrainCameras().copy()
        if not train_cams:
            return

        best = []
        with torch.no_grad():
            for _ in range(min(3, len(train_cams))):
                cam = train_cams[np.random.randint(0, len(train_cams))]
                pkg = render(cam, self, pipe, background)
                score = (pkg["rend_alpha"].squeeze() < opt.alpha_hole).sum().item()
                best.append((score, cam))
            best.sort(reverse=True, key=lambda x: x[0])

        if len(best) == 0:
            if num_to_prune > 0:
                print(f"[Topo] Pruning {num_to_prune} points.")
                self.prune_points(prune_mask, iteration)
            return

        union_hole_mask = None
        depth_pack = []
        with torch.no_grad():
            for _, cam in best:
                pkg = render(cam, self, pipe, background)
                a = pkg["rend_alpha"].squeeze()
                d = pkg["surf_depth"].squeeze()
                m = (a < opt.alpha_hole)
                union_hole_mask = m if union_hole_mask is None else (union_hole_mask | m)
                depth_pack.append((m, d, cam))

        if union_hole_mask is None or not union_hole_mask.any():
            if num_to_prune > 0:
                print(f"[Topo] Pruning {num_to_prune} points.")
                self.prune_points(prune_mask, iteration)
            return

        new_plans = []
        for m, depth, cam in depth_pack:
            if not m.any():
                continue
            dil = kornia.morphology.dilation(m[None, None].float(), torch.ones(3, 3, device="cuda")).squeeze().bool()
            boundary = dil & ~m
            if not boundary.any():
                continue

            val = depth[boundary]
            lo, hi = torch.quantile(val, 0.01), torch.quantile(val, 0.99)
            ok = (val > lo) & (val < hi)
            if ok.sum() == 0:
                continue
            by, bx = torch.nonzero(boundary, as_tuple=True)
            by, bx = by[ok], bx[ok]
            val = val[ok]

            K = torch.tensor(cam.K, device="cuda", dtype=torch.float32)
            c2w = torch.inverse(cam.world_view_transform).cuda()
            x = (bx - K[0, 2]) * val / K[0, 0]
            y = (by - K[1, 2]) * val / K[1, 1]
            z = val
            pts_cam = torch.stack([x, y, z], dim=-1)
            pts_homo = torch.cat([pts_cam, torch.ones_like(z.unsqueeze(-1))], dim=-1)
            xyz_world = (c2w @ pts_homo.T).T[:, :3]

            kdt = cKDTree(self._xyz.detach().cpu().numpy())
            _, nn_idx_np = kdt.query(xyz_world.cpu().numpy(), k=1)
            nn_idx = torch.from_numpy(nn_idx_np).long().cuda()

            boundary_labels = comp_labels[nn_idx]
            uniq, cnt = torch.unique(boundary_labels, return_counts=True)
            sig_on_boundary = uniq[sig_mask[uniq]]

            if len(sig_on_boundary) > 1:
                counts = cnt[torch.isin(uniq, sig_on_boundary)]
                if counts.numel() >= 2:
                    top2 = torch.topk(counts, k=2).indices
                    l1 = sig_on_boundary[top2[0]]
                    l2 = sig_on_boundary[top2[1]]
                    pts1 = self._xyz[nn_idx[boundary_labels == l1]]
                    pts2 = self._xyz[nn_idx[boundary_labels == l2]]

                    gu = _gu()
                    R = gu.build_rotation(self._rotation)
                    normals = R[..., 2]
                    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
                        continue
                    n1 = F.normalize(normals[nn_idx[boundary_labels == l1]].mean(dim=0), p=2, dim=0)
                    n2 = F.normalize(normals[nn_idx[boundary_labels == l2]].mean(dim=0), p=2, dim=0)
                    dot = torch.clamp(torch.sum(n1 * n2), -1.0, 1.0).abs()
                    angle_rad = torch.acos(dot)
                    angle_deg = float(torch.rad2deg(angle_rad))
                    if angle_deg > 30.0:
                        continue

                    brid = self._create_bridging_gaussians(pts1, pts2, opt.lbo_bridge_points_count)
                    if isinstance(brid, dict) and 'xyz' in brid and brid['xyz'].shape[0] > 0:
                        new_plans.append(brid)
            elif len(sig_on_boundary) == 1:
                label = sig_on_boundary[0]
                idxs = nn_idx[boundary_labels == label]
                if idxs.shape[0] > opt.lbo_planning_max_new_points:
                    perm = torch.randperm(idxs.shape[0], device="cuda")
                    idxs = idxs[perm[:opt.lbo_planning_max_new_points]]
                fill = self._create_filling_gaussians(idxs, k=8)
                if fill:
                    new_plans.append(fill)

        if num_to_prune > 0:
            print(f"[Topo] Pruning {num_to_prune} points.")
            self.prune_points(prune_mask, iteration)

        if len(new_plans) > 0:
            all_xyz_list = [d['xyz'] for d in new_plans if 'xyz' in d]
            all_xyz = torch.cat(all_xyz_list, dim=0) if len(all_xyz_list) > 0 else torch.empty((0,3), device="cuda")

            rot_list = []
            for d in new_plans:
                if 'xyz' in d:
                    if 'rotation' in d and d['rotation'] is not None:
                        rot_list.append(d['rotation'])
                    else:
                        rot_list.append(None)

            if all_xyz.shape[0] > 0:
                print(f"[Topo] Adding {all_xyz.shape[0]} new Gaussians.")
                kdt = cKDTree(self.get_xyz.detach().cpu().numpy())
                _, nn = kdt.query(all_xyz.detach().cpu().numpy(), k=1)
                nn = torch.from_numpy(nn).long().cuda()

                new_f_dc = self._features_dc[nn]
                new_f_rest = self._features_rest[nn]
                new_scal = self._scaling[nn]
                new_opac = self.inverse_opacity_activation(torch.full((all_xyz.shape[0], 1), 0.02, device="cuda"))
                new_rot = self._rotation[nn].clone()

                if any(r is not None for r in rot_list):
                    cursor = 0
                    for idx, d in enumerate(new_plans):
                        if 'xyz' not in d:
                            continue
                        seg_len = d['xyz'].shape[0]
                        if 'rotation' in d and d['rotation'] is not None:
                            new_rot[cursor:cursor+seg_len] = d['rotation']
                        cursor += seg_len

                self.densification_postfix(
                    all_xyz, new_f_dc, new_f_rest, new_opac, new_scal, new_rot, all_xyz.clone(), iteration
                )

        torch.cuda.empty_cache()

    @staticmethod
    def combine_gslist(gslist):
        """
        Combine a list of GaussianModel objects into a single GaussianModel object.
        
        Args:
            gslist: List of GaussianModel objects to combine
            
        Returns:
            A new GaussianModel instance containing all parameters from the input models
        """
        if not gslist:
            raise ValueError("gslist cannot be empty")
        combined_model = GaussianModel(gslist[0].max_sh_degree)
        
        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        mip_filter_list = []
        origin_xyz_list = []
        creation_iter_list = []
        is_protected_list = []

        for model in gslist:
            xyz_list.append(model.get_xyz.detach())
            features_dc_list.append(model._features_dc.detach())
            features_rest_list.append(model._features_rest.detach())
            opacity_list.append(model._opacity.detach())
            scaling_list.append(model._scaling.detach())
            rotation_list.append(model._rotation.detach())
            origin_xyz_list.append(model.origin_xyz.detach())
            creation_iter_list.append(model._creation_iter.detach() if model._creation_iter.numel() > 0 else torch.zeros_like(model.get_xyz.detach()[:, 0], dtype=torch.long))
            is_protected_list.append(model._is_protected.detach() if model._is_protected.numel() > 0 else torch.zeros_like(model.get_xyz.detach()[:, 0], dtype=torch.bool))

            if hasattr(model, "use_mip_filter") and model.use_mip_filter and hasattr(model, "mip_filter"):
                mip_filter_list.append(model.mip_filter.detach())
            else:
                mip_filter_list.append(torch.zeros((model.get_xyz.shape[0], 1), device=model.get_xyz.device))
        
        combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
        combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
        combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
        combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
        combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
        combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))
        combined_model.origin_xyz = torch.cat(origin_xyz_list, dim=0)
        combined_model._creation_iter = torch.cat(creation_iter_list, dim=0)
        combined_model._is_protected = torch.cat(is_protected_list, dim=0)

        combined_model.use_mip_filter = any(model.use_mip_filter for model in gslist)
        if combined_model.use_mip_filter:
            combined_model.mip_filter = torch.cat(mip_filter_list, dim=0)
        else:
            combined_model.mip_filter = torch.empty(0)
        
        combined_model.active_sh_degree = gslist[0].active_sh_degree
        n_points = combined_model._xyz.shape[0]
        combined_model.max_radii2D = torch.zeros(n_points, device=combined_model._xyz.device)
        combined_model.xyz_gradient_accum = torch.zeros((n_points, 1), device=combined_model._xyz.device)
        combined_model.denom = torch.zeros((n_points, 1), device=combined_model._xyz.device)
        combined_model.percent_dense = gslist[0].percent_dense
        combined_model.spatial_lr_scale = gslist[0].spatial_lr_scale
        combined_model._denom_prune_min = gslist[0]._denom_prune_min

        first_model_opt = gslist[0].optimizer
        lr_config = {}
        for group in first_model_opt.param_groups:
            lr_config[group["name"]] = group["lr"]
        
        l = [
            {'params': [combined_model._xyz], 'lr': lr_config.get("xyz", 0.001), "name": "xyz"},
            {'params': [combined_model._features_dc], 'lr': lr_config.get("f_dc", 0.001), "name": "f_dc"},
            {'params': [combined_model._features_rest], 'lr': lr_config.get("f_rest", 0.001/20), "name": "f_rest"},
            {'params': [combined_model._opacity], 'lr': lr_config.get("opacity", 0.001), "name": "opacity"},
            {'params': [combined_model._scaling], 'lr': lr_config.get("scaling", 0.001), "name": "scaling"},
            {'params': [combined_model._rotation], 'lr': lr_config.get("rotation", 0.001), "name": "rotation"}
        ]
        combined_model.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        print(f"Combined {len(gslist)} models with a total of {n_points} points")
        for i, model in enumerate(gslist):
            print(f"  Model {i}: {model.get_xyz.shape[0]} points")
        
        return combined_model

    @staticmethod
    def combine_gslist_simple(gslist):
        if not gslist:
            raise ValueError("gslist cannot be empty")
        combined_model = GaussianModel(gslist[0].max_sh_degree)
        
        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        origin_xyz_list = []

        for model in gslist:
            xyz_list.append(model.get_xyz.detach())
            features_dc_list.append(model._features_dc.detach())
            features_rest_list.append(model._features_rest.detach())
            opacity_list.append(model._opacity.detach())
            scaling_list.append(model._scaling.detach())
            rotation_list.append(model._rotation.detach())
            origin_xyz_list.append(model.origin_xyz.detach() if hasattr(model, "origin_xyz") else model.get_xyz.detach().clone())

        combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
        combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
        combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
        combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
        combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
        combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))
        combined_model.origin_xyz = torch.cat(origin_xyz_list, dim=0)

        combined_model.max_radii2D = torch.zeros(combined_model._xyz.shape[0], device=combined_model._xyz.device)
        combined_model.xyz_gradient_accum = torch.zeros((combined_model._xyz.shape[0], 1), device=combined_model._xyz.device)
        combined_model.denom = torch.zeros((combined_model._xyz.shape[0], 1), device=combined_model._xyz.device)
        combined_model.active_sh_degree = gslist[0].active_sh_degree
        combined_model.percent_dense = gslist[0].percent_dense
        combined_model.spatial_lr_scale = gslist[0].spatial_lr_scale
        combined_model._denom_prune_min = gslist[0]._denom_prune_min

        combined_model.use_mip_filter = any(getattr(model, "use_mip_filter", False) for model in gslist)
        if combined_model.use_mip_filter:
            mip_filter_list = []
            for model in gslist:
                if hasattr(model, "mip_filter") and model.mip_filter.numel() > 0:
                    mip_filter_list.append(model.mip_filter.detach())
                else:
                    mip_filter_list.append(torch.zeros((model.get_xyz.shape[0], 1), device=model.get_xyz.device))
            combined_model.mip_filter = torch.cat(mip_filter_list, dim=0)
        else:
            combined_model.mip_filter = torch.empty(0)

        gaussians_num = len(gslist)
        total_count = combined_model._xyz.shape[0]
        print(f'{gaussians_num} Gaussians are combined into one model, with {total_count} points')

        return combined_model

    @staticmethod
    def get_obj_gaussian_by_mask(gaussian, obj_gs_mask):
        if not isinstance(gaussian, GaussianModel):
            raise TypeError("gaussian must be an instance of GaussianModel")
        if obj_gs_mask.shape[0] != gaussian.get_xyz.shape[0]:
            raise ValueError("obj_gs_mask shape must match gaussian points count")
        
        obj_gaussian = GaussianModel(gaussian.max_sh_degree)
        
        obj_gaussian._xyz = torch.nn.Parameter(gaussian._xyz[obj_gs_mask].clone().detach())
        obj_gaussian._features_dc = torch.nn.Parameter(gaussian._features_dc[obj_gs_mask].clone().detach())
        obj_gaussian._features_rest = torch.nn.Parameter(gaussian._features_rest[obj_gs_mask].clone().detach())
        obj_gaussian._scaling = torch.nn.Parameter(gaussian._scaling[obj_gs_mask].clone().detach())
        obj_gaussian._rotation = torch.nn.Parameter(gaussian._rotation[obj_gs_mask].clone().detach())
        obj_gaussian._opacity = torch.nn.Parameter(gaussian._opacity[obj_gs_mask].clone().detach())
        
        if hasattr(gaussian, "origin_xyz"):
            obj_gaussian.origin_xyz = gaussian.origin_xyz[obj_gs_mask].clone().detach()
        else:
            obj_gaussian.origin_xyz = obj_gaussian._xyz.clone().detach()
        
        if hasattr(gaussian, "_creation_iter") and gaussian._creation_iter.numel() > 0:
            obj_gaussian._creation_iter = gaussian._creation_iter[obj_gs_mask].clone().detach()
        else:
            obj_gaussian._creation_iter = torch.zeros(obj_gaussian._xyz.shape[0], dtype=torch.long, device=obj_gaussian._xyz.device)
        
        if hasattr(gaussian, "_is_protected") and gaussian._is_protected.numel() > 0:
            obj_gaussian._is_protected = gaussian._is_protected[obj_gs_mask].clone().detach()
        else:
            obj_gaussian._is_protected = torch.zeros(obj_gaussian._xyz.shape[0], dtype=torch.bool, device=obj_gaussian._xyz.device)

        obj_gaussian.active_sh_degree = gaussian.active_sh_degree
        obj_gaussian.max_sh_degree = gaussian.max_sh_degree
        obj_gaussian.spatial_lr_scale = gaussian.spatial_lr_scale
        obj_gaussian.percent_dense = gaussian.percent_dense
        obj_gaussian._denom_prune_min = gaussian._denom_prune_min

        obj_gaussian.use_mip_filter = gaussian.use_mip_filter
        if obj_gaussian.use_mip_filter:
            if hasattr(gaussian, 'mip_filter') and gaussian.mip_filter.numel() > 0:
                obj_gaussian.mip_filter = gaussian.mip_filter[obj_gs_mask].clone().detach()
            else:
                obj_gaussian.mip_filter = torch.zeros((obj_gaussian._xyz.shape[0], 1), device=obj_gaussian._xyz.device)
        else:
            obj_gaussian.mip_filter = torch.empty(0)

        obj_gaussian.max_radii2D = torch.zeros(obj_gaussian._xyz.shape[0], device=obj_gaussian._xyz.device)
        obj_gaussian.xyz_gradient_accum = torch.zeros((obj_gaussian._xyz.shape[0], 1), device=obj_gaussian._xyz.device)
        obj_gaussian.denom = torch.zeros((obj_gaussian._xyz.shape[0], 1), device=obj_gaussian._xyz.device)

        print(f"Extracted {obj_gaussian._xyz.shape[0]} Gaussians for the object "
              f"(out of {gaussian._xyz.shape[0]} total Gaussians)")
        
        return obj_gaussian

    @staticmethod
    def get_gaussian_normal(rotation, scaling, scale_modifier=1.0):
        if rotation.dim() != 2 or rotation.shape[-1] != 4:
            raise ValueError("rotation must be (N,4) quaternion tensor")
        if scaling.dim() != 2 or scaling.shape[-1] != 2:
            raise ValueError("scaling must be (N,2) 2D scaling tensor")
        
        gu = _gu()
        q = torch.nn.functional.normalize(rotation, dim=-1)
        scales_3d = torch.cat([scaling * scale_modifier, torch.ones_like(scaling[:, :1])], dim=-1)
        L = gu.build_scaling_rotation(scales_3d, q)  # (N, 3, 3), L = R * S
        normal = L[:, :, 2]
        return normal
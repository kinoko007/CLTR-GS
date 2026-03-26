import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import argparse
import json
import time
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command_safe(command: str):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    print("Command succeeded!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)

    parser.add_argument('--n_images', type=int, default=None)
    parser.add_argument('--use_view_config', action='store_true')
    parser.add_argument('--config_view_num', type=int, default=10)
    parser.add_argument('--image_idx', type=int, nargs='*', default=None)
    parser.add_argument('--randomize_images', action='store_true')

    parser.add_argument('--dense_supervision', action='store_true')
    parser.add_argument('--dense_regul', type=str, default='default')

    parser.add_argument('--use_multires_tsdf', action='store_true')
    parser.add_argument('--no_interpolated_views', action='store_true')

    parser.add_argument('--sfm_config', type=str, default='unposed')

    parser.add_argument('--alignment_config', type=str, default='default')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')

    parser.add_argument('--free_gaussians_config', type=str, default=None)

    parser.add_argument('--tsdf_config', type=str, default='default')
    parser.add_argument('--tetra_config', type=str, default='default')
    parser.add_argument('--tetra_downsample_ratio', type=float, default=0.5)

    parser.add_argument('--select_inpaint_num', type=int, default=20)
    parser.add_argument('--use_downsample_gaussians', action='store_true')
    parser.add_argument('--use_mesh_filter', action='store_true')
    parser.add_argument('--use_dense_view', action='store_true')

    parser.add_argument("--curriculum_rot_deg_s1", type=float, default=5)# 10
    parser.add_argument("--curriculum_rot_deg_s2", type=float, default=10)# 20
    parser.add_argument("--curriculum_rot_deg_s3", type=float, default=15)# 30
    parser.add_argument("--curriculum_rot_deg", type=float, default=None)

    args = parser.parse_args()

    if args.output_path is None:
        output_dir_name = args.source_path.split(os.sep)[-2] if args.source_path.endswith(os.sep) else args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)

    mast3r_scene_path = os.path.join(args.output_path, 'mast3r_sfm')
    aligned_charts_path = os.path.join(args.output_path, 'mast3r_sfm')
    free_gaussians_path = os.path.join(args.output_path, 'free_gaussians')
    tsdf_meshes_path = os.path.join(args.output_path, 'tsdf_meshes')
    tetra_meshes_path = os.path.join(args.output_path, 'tetra_meshes')

    if args.use_dense_view:
        dense_view_json_path = os.path.join(args.source_path, 'dense_view.json')
        print(f'[INFO]: Search dense view json in {dense_view_json_path}')
        if not os.path.exists(dense_view_json_path):
            source_img_path = os.path.join(args.source_path, 'images')
            source_img_num = len(os.listdir(source_img_path))
            with open(dense_view_json_path, 'w') as f:
                json.dump({'train': list(range(source_img_num))}, f)
            print(f"[INFO] Save dense view index list to {dense_view_json_path}")

    dense_arg = ""

    if args.free_gaussians_config is None:
        args.free_gaussians_config = 'long' if args.dense_supervision else 'default'

    if args.use_view_config:
        n_images = None
        view_config_path = os.path.join(args.source_path, f'split-{args.config_view_num}views.json')
        if os.path.exists(view_config_path):
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train']
        else:
            view_config_path = os.path.join(args.source_path, f'train_test_split_{args.config_view_num}.json')
            with open(view_config_path, 'r') as f:
                view_config = json.load(f)
            image_idx_list = view_config['train_ids']
    else:
        n_images = args.n_images
        image_idx_list = args.image_idx

    sfm_command = " ".join([
        "python", "scripts/run_sfm.py",
        "--source_path", args.source_path,
        "--output_path", mast3r_scene_path,
        "--config", args.sfm_config,
        "--n_images" if n_images is not None else "", str(n_images) if n_images is not None else "",
        "--image_idx" if image_idx_list is not None else "", " ".join([str(i) for i in image_idx_list]) if image_idx_list is not None else "",
        "--randomize_images" if args.randomize_images else "",
    ])

    align_charts_command = " ".join([
        "python", "scripts/align_charts.py",
        "--source_path", mast3r_scene_path,
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", aligned_charts_path,
        "--config", args.alignment_config,
        "--depth_model", args.depth_model,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
    ])

    plane_root_path = os.path.join(mast3r_scene_path, 'plane-refine-depths')

    refine_free_gaussians_command = " ".join([
        "python", "scripts/refine_free_gaussians.py",
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", free_gaussians_path,
        "--config", args.free_gaussians_config,
        dense_arg,
        "--dense_regul", args.dense_regul,
        "--refine_depth_path", plane_root_path,
        "--use_lbo_planning", "False",
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
    ])

    refine_free_gaussians_command_stage3 = " ".join([
        "python", "scripts/refine_free_gaussians.py",
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", free_gaussians_path,
        "--config", args.free_gaussians_config,
        dense_arg,
        "--dense_regul", args.dense_regul,
        "--refine_depth_path", plane_root_path,
        "--use_lbo_planning", "True",
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
    ])

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
        "--use_default_output_dir",
    ])

    tsdf_command = " ".join([
        "python", "scripts/extract_tsdf_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tsdf_meshes_path,
        "--config", args.tsdf_config,
    ])

    tetra_command = " ".join([
        "python", "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views" if not args.no_interpolated_views else "",
        dense_arg,
    ])

    def get_see3d_inpaint_command(stage, select_inpaint_num):
        parts = [
            "python", "scripts/see3d_inpaint.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--plane_root_dir", plane_root_path,
            "--iteration", "7000",
            "--see3d_stage", str(stage),
            "--select_inpaint_num", str(select_inpaint_num),
        ]
        if args.curriculum_rot_deg_s1 is not None:
            parts += ["--curriculum_rot_deg_s1", str(args.curriculum_rot_deg_s1)]
        if args.curriculum_rot_deg_s2 is not None:
            parts += ["--curriculum_rot_deg_s2", str(args.curriculum_rot_deg_s2)]
        if args.curriculum_rot_deg_s3 is not None:
            parts += ["--curriculum_rot_deg_s3", str(args.curriculum_rot_deg_s3)]
        if args.curriculum_rot_deg is not None:
            parts += ["--curriculum_rot_deg", str(args.curriculum_rot_deg)]
        return " ".join(parts)

    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", args.source_path,
        "--model_path", args.output_path,
        "--sparse_view_num", str(args.config_view_num),
    ])

    render_charts_command = " ".join([
        "python", "2d-gaussian-splatting/render_chart_views.py",
        "--source_path", mast3r_scene_path,
        "--save_root_path", plane_root_path,
    ])

    generate_2Dplane_command = " ".join([
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", plane_root_path,
    ])

    pnts_path = os.path.join(mast3r_scene_path, 'chart_pcd.ply')

    def get_plane_refine_depth_command(anchor_view_id_json_path=None, see3d_root_path=None):
        if see3d_root_path is not None:
            if anchor_view_id_json_path is not None:
                return " ".join([
                    "python", "scripts/plane_refine_depth.py",
                    "--source_path", mast3r_scene_path,
                    "--plane_root_path", plane_root_path,
                    "--pnts_path", pnts_path,
                    "--anchor_view_id_json_path", anchor_view_id_json_path,
                    "--see3d_root_path", see3d_root_path,
                ])
            return " ".join([
                "python", "scripts/plane_refine_depth.py",
                "--source_path", mast3r_scene_path,
                "--plane_root_path", plane_root_path,
                "--pnts_path", pnts_path,
                "--see3d_root_path", see3d_root_path,
            ])
        return " ".join([
            "python", "scripts/plane_refine_depth.py",
            "--source_path", mast3r_scene_path,
            "--plane_root_path", plane_root_path,
            "--pnts_path", pnts_path,
        ])

    see3d_root_path = os.path.join(mast3r_scene_path, 'see3d_render')

    t1 = time.time()

    run_command_safe(sfm_command)
    run_command_safe(align_charts_command)

    run_command_safe(render_charts_command)
    run_command_safe(generate_2Dplane_command)
    run_command_safe(get_plane_refine_depth_command(None, None))

    run_command_safe(refine_free_gaussians_command)

    if args.use_dense_view:
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.bin', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.bin')
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.txt', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.txt')
        shutil.copy(f'{mast3r_scene_path}/sparse/0/points3D.ply', f'{mast3r_scene_path}/dense-view-sparse/0/points3D.ply')

        run_command_safe(" ".join([
            "python", "2d-gaussian-splatting/render_dense_views.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--iteration", "7000",
        ]))

        run_command_safe(" ".join([
            "python", "2d-gaussian-splatting/guidance/dense_dn_util.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--iteration", "7000",
        ]))

        run_command_safe(generate_2Dplane_command)
        run_command_safe(get_plane_refine_depth_command(None, None))
        run_command_safe(f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-chart-views')

        run_command_safe(refine_free_gaussians_command)

        run_command_safe(render_all_img_command)
        run_command_safe(tetra_command)

        print("Finished training dense view without See3D prior!")
        print(f"Total running time: {time.time() - t1} seconds")
        sys.exit(0)

    # stage1
    run_command_safe(get_see3d_inpaint_command(1, args.select_inpaint_num))
    run_command_safe(get_plane_refine_depth_command(None, see3d_root_path))
    run_command_safe(f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-ori')
    run_command_safe(refine_free_gaussians_command)

    # stage2
    run_command_safe(get_see3d_inpaint_command(2, args.select_inpaint_num))
    run_command_safe(get_plane_refine_depth_command(None, see3d_root_path))
    run_command_safe(f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s1')
    run_command_safe(refine_free_gaussians_command)

    # stage3
    run_command_safe(get_see3d_inpaint_command(3, args.select_inpaint_num))
    anchor_view_id_json_path = os.path.join(see3d_root_path, 'stage3', 'anchor_view_id.json')
    run_command_safe(get_plane_refine_depth_command(anchor_view_id_json_path, see3d_root_path))
    run_command_safe(f'mv {free_gaussians_path}/point_cloud {free_gaussians_path}/point_cloud-s2')

    run_command_safe(refine_free_gaussians_command_stage3)

    run_command_safe(render_all_img_command)
    run_command_safe(tetra_command)

    if args.use_mesh_filter:
        mesh_path = os.path.join(tetra_meshes_path, 'tetra_mesh_binary_search_7_iter_7000.ply')
        length_threshold = 0.5
        filtered_mesh_path = os.path.join(tetra_meshes_path, f'tetra_mesh_binary_search_7_iter_7000_filtered_t{length_threshold}.ply')

        run_command_safe(" ".join([
            "python", "2d-gaussian-splatting/utils/mesh_filter.py",
            "--mesh_path", mesh_path,
            "--output_path", filtered_mesh_path,
        ]))

        run_command_safe(f'mv {mesh_path} {tetra_meshes_path}/tetra_mesh_binary_search_7_iter_7000_ori.ply')
        run_command_safe(f'mv {filtered_mesh_path} {mesh_path}')

    run_command_safe(eval_command)

    print(f"Total running time: {time.time() - t1} seconds")

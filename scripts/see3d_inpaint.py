import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import subprocess


def run_command_safe(cmd_list):
    printable = " ".join([str(x) for x in cmd_list])
    print(f"Running command: {printable}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed! return_code={e.returncode}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Command failed! {e}")
        sys.exit(1)
    print("Command succeeded!")

def pick_stage_rot_deg(stage: int, args):
    """
    stage-specific > global > None(=do not override)
    """
    if stage == 1 and args.curriculum_rot_deg_s1 is not None:
        return args.curriculum_rot_deg_s1
    if stage == 2 and args.curriculum_rot_deg_s2 is not None:
        return args.curriculum_rot_deg_s2
    if stage == 3 and args.curriculum_rot_deg_s3 is not None:
        return args.curriculum_rot_deg_s3
    return args.curriculum_rot_deg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--plane_root_dir", type=str, required=True)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--select_inpaint_num", required=True, type=str)

    parser.add_argument("--curriculum_rot_deg_s1", type=float, default=None)
    parser.add_argument("--curriculum_rot_deg_s2", type=float, default=None)
    parser.add_argument("--curriculum_rot_deg_s3", type=float, default=None)
    parser.add_argument("--curriculum_rot_deg", type=float, default=None)

    parser.add_argument("--no_delete_unselected_views", action="store_true",
                        help="Do NOT delete unselected novel-view files under raw-gs/.")

    args = parser.parse_args()

    yaw_deg = pick_stage_rot_deg(args.see3d_stage, args)

    # 1) render novel views
    cmd = [
        "python", "2d-gaussian-splatting/render_novel_views.py",
        "--source_path", args.source_path,
        "--model_path", args.model_path,
        "--iteration", str(args.iteration),
        "--see3d_stage", str(args.see3d_stage),
        "--select_inpaint_num", str(args.select_inpaint_num),
    ]
    if yaw_deg is not None:
        cmd += ["--curriculum_rot_deg", str(yaw_deg)]
    if args.no_delete_unselected_views:
        cmd += ["--no_delete_unselected_views"]

    run_command_safe(cmd)

    ref_image_path = os.path.join(args.source_path, 'see3d_render', 'ref-views')
    warp_image_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs')
    output_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-inpainted')

    # 2) inpaint rgb
    cmd = [
        "python", "2d-gaussian-splatting/guidance/see3d_util.py",
        "--ref_imgs_dir", ref_image_path,
        "--warp_root_dir", warp_image_path,
        "--output_root_dir", output_root_dir,
    ]
    run_command_safe(cmd)

    # 3) generate depth and normal
    cmd = [
        "python", "2d-gaussian-splatting/guidance/see3d_dn_util.py",
        "--source_path", args.source_path,
        "--see3d_stage", str(args.see3d_stage),
    ]
    run_command_safe(cmd)

    # 4) generate 2D planes
    cur_plane_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-planes')
    cmd = [
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", cur_plane_root_dir,
    ]
    run_command_safe(cmd)

    # 5) merge results
    anchor_view_id_json_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'anchor_view_id.json')
    cmd = [
        "python", "2d-gaussian-splatting/guidance/merge_util.py",
        "--source_path", args.source_path,
        "--see3d_stage", str(args.see3d_stage),
        "--plane_root_dir", args.plane_root_dir,
        "--anchor_view_id_json_path", anchor_view_id_json_path,
        "--none_replace",
    ]
    run_command_safe(cmd)

    print(f'See3D stage {args.see3d_stage} inpaint done!')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from rich.console import Console

def run_command_safe(command: str):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")


def str2bool(v):
    """
    Compatible bool parser:
      - --flag            -> True (via const=True)
      - --flag True/False -> True/False
      - --flag 1/0        -> True/False
      - --flag yes/no     -> True/False
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)

    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}. Use True/False, 1/0, yes/no.")


def join_cmd(parts):
    """Join command parts while filtering empty strings."""
    return " ".join([p for p in parts if p is not None and str(p).strip() != ""])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True, help='Path to MASt3R scene (SfM output root)')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Output directory for refined gaussians')

    parser.add_argument(
        '--white_background',
        default=False,
        nargs='?',
        const=True,
        type=str2bool,
        help='Use white background. Support: --white_background or --white_background True/False'
    )

    # For dense RGB and depth supervision from a COLMAP dataset (optional)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default',
                        help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')

    # Config
    parser.add_argument('-c', '--config', type=str, default='default')

    parser.add_argument('--refine_depth_path', type=str, default=None, help='Path to the refine depth directory')
    parser.add_argument('--use_downsample_gaussians', action='store_true', help='Use downsample gaussians')

    parser.add_argument(
        '--use_lbo_planning',
        default=False,
        nargs='?',
        const=True,
        type=str2bool,
        help='Enable LBO planning. Support: --use_lbo_planning or --use_lbo_planning True/False'
    )

    args = parser.parse_args()

    # Set console
    CONSOLE = Console(width=120)

    # Set output path
    if args.output_path is None:
        scene_path = args.mast3r_scene
        scene_path = scene_path[:-1] if scene_path.endswith(os.sep) else scene_path
        output_dir_name = os.path.basename(scene_path)
        args.output_path = os.path.join('output', output_dir_name, 'refined_free_gaussians')

    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Refined free gaussians will be saved to: {args.output_path}")

    # Load config
    config_path = os.path.join('configs/free_gaussians_refinement', args.config + '.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Dense supervision (optional)
    dense_arg = ""
    if args.dense_data_path is not None:
        dense_arg = join_cmd(["--dense_data_path", args.dense_data_path])

    # Define command
    if args.refine_depth_path is None:
        raise ValueError('refine depth path is required (--refine_depth_path)')

    CONSOLE.print(f"[INFO] refine depth path: {args.refine_depth_path}, train GS will use refine depth")

    command_parts = [
        "python", "2d-gaussian-splatting/train_with_refine_depth.py",
        "-s", args.mast3r_scene,
        "-m", args.output_path,
        "--iterations", str(config.get('iterations')),
        "--densify_until_iter", str(config.get('densify_until_iter')),
        "--opacity_reset_interval", str(config.get('opacity_reset_interval')),
        "--depth_ratio", str(config.get('depth_ratio')),
        "--use_mip_filter" if config.get('use_mip_filter', False) else "",
        dense_arg,
        "--normal_consistency_from", str(config.get('normal_consistency_from')),
        "--distortion_from", str(config.get('distortion_from')),
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
        "--dense_regul", args.dense_regul,
        "--refine_depth_path", args.refine_depth_path,
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
        "--white_background", str(bool(args.white_background)),
        "--use_lbo_planning", str(bool(args.use_lbo_planning)),
    ]

    command = join_cmd(command_parts)

    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{command}")
    run_command_safe(command)

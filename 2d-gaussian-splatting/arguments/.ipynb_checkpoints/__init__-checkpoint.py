# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


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
    raise ValueError(f"Invalid boolean value: {v}. Use True/False, 1/0, yes/no.")


class ParamGroup:
    """
    Compatible ParamGroup:
    - bool args support both:
        --foo
        --foo True/False
    - non-bool args keep normal behavior
    """
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            default_value = value if not fill_none else None

            if t == bool:
                if shorthand:
                    group.add_argument(
                        "--" + key, "-" + key[0:1],
                        default=default_value,
                        nargs="?",
                        const=True,
                        type=str2bool
                    )
                else:
                    group.add_argument(
                        "--" + key,
                        default=default_value,
                        nargs="?",
                        const=True,
                        type=str2bool
                    )
                continue

            if shorthand:
                group.add_argument("--" + key, "-" + key[0:1], default=default_value, type=t)
            else:
                group.add_argument("--" + key, default=default_value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.denom_prune_min = 5

        # Topology (2DGS)
        self.use_lbo_planning = False
        self.lbo_planning_start_iter = 0
        self.lbo_planning_interval = 1000
        self.lbo_planning_max_new_points = 100000
        self.lbo_planning_min_component_size = 2000
        self.lbo_bridge_points_count = 5000
        self.grace_period = 0
        self.alpha_hole = 0.3

        # Memory-friendly KNN/CC
        self.knn_k = 8 #8->4/16
        self.topo_candidate_k = 32
        self.topo_max_points = 10000_0000
        self.knn_refine_on_gpu = True

        self.freeze_new = 0

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

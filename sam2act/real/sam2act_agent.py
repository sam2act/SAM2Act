import os
import logging

import torch

from robits.agents.base_agent import BaseAgent

import sam2act.mvt.config as default_mvt_cfg
import sam2act.models.sam2act_agent as sam2act_agent
import sam2act.config as default_exp_cfg
import sam2act.mvt.mvt_sam2 as mvt_sam2

from sam2act.utils.rvt_utils import load_agent_only_model as load_agent_state

from sam2act.eval import get_model_size

logger = logging.getLogger(__name__)


CAMERAS_REAL = ["front"]
SCENE_BOUNDS_REAL = [
    0.0,
    -0.5,
    0.0,
    1.0,
    0.5,
    1.0,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE_REAL_WIDTH = 640
IMAGE_SIZE_REAL_HEIGHT = 480


class SAM2Act(BaseAgent):

    def __init__(self, model_path):
        self.device = torch.device("cuda:0")

        self.agent = load_agent(model_path, self.device)
        self.agent.load_clip()
        self.agent.eval()
        self.lang_goal = "push the buttons in the following order: red, green, blue"

    @torch.inference_mode()
    def get_action(self, step, observation):
        if not observation:
            logger.error("Nothing todo.")
            return

        with torch.jit.optimized_execution(False):
            act_result = self.agent.act(step, observation, deterministic=True)
        return act_result


def load_agent(model_path: str, device):
    """
    .. seealso:: sam2act.eval:load_agent
    """
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    mvt_cfg = default_mvt_cfg.get_cfg_defaults()

    if "_plus_" not in model_path:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg_plus.yaml"))
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg_plus.yaml"))

    exp_cfg.freeze()
    mvt_cfg.freeze()

    sam2act = mvt_sam2.MVT_SAM2(
        renderer_device=device,
        rank=0,
        **mvt_cfg,
    )

    get_model_size(sam2act)

    agent = sam2act_agent.SAM2Act_Agent(
        network=sam2act.to(device),
        image_resolution=[IMAGE_SIZE_REAL_WIDTH, IMAGE_SIZE_REAL_HEIGHT],
        add_lang=mvt_cfg.add_lang,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_REAL,
        cameras=CAMERAS_REAL,
        log_dir="./eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()

    return agent

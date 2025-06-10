#!/usr/bin/env python3

import sys
import logging
import time

import numpy as np

import torch

import questionary
import rich_click as click
from click_prompt import filepath_option
from click_prompt import choice_option


from robits.core.abc.control import control_types

from robits.core.abc.robot import UnimanualRobot as Robot
from robits.core.data_model.action import CartesianAction
from robits.core.data_model.camera_capture import CameraData
from robits.vis.scene_visualizer import SceneVisualizer

from robits.cli import cli_options
from robits.cli import cli_utils

from sam2act.real.sam2act_agent import SAM2Act as Agent

logger = logging.getLogger(__name__)

execution_mode = ["step", "auto", "vis"]

used_intructions = [
    "push the red button and then the blue button",
    "close and open the same drawer",
    "put the blue in drawer",
    "put the orange in drawer",
]


arm_actions = ["execute", "skip", "quit"]


def execute_action(robot: Robot, action: CartesianAction):
    with robot.control(control_types.cartesian) as ctrl:
        ctrl.update((action.position, action.quaternion))

    if action.hand_open:
        robot.gripper.open()
    else:
        robot.gripper.close()


def prompt_for_action(robot, action: CartesianAction):

    action_mode = questionary.select(
        "What should I do with the arms ?",
        arm_actions,
        default="skip",
        use_shortcuts=True,
    ).ask()

    if action_mode == "quit":
        sys.exit(0)

    elif action_mode == "execute":
        execute_action(robot, action)
        return

    elif action_mode == "skip":
        return


@torch.inference_mode()
@click.command()
@filepath_option(
    "--model-path",
    default="/home/markus/sam2act_models/",
    help="Path to the model file",
)
@choice_option(
    "--execution-mode", type=click.Choice(execution_mode), help="Execution mode"
)
@cli_options.robot()
@click.option(
    "--use-audio/--no-audio",
    default=False,
    is_flag=True,
    help="Get the language command from audio",
)
@choice_option(
    "--instruction",
    type=click.Choice(used_intructions),
    help="Specify the language command that is passed to the model",
)
def cli(model_path, execution_mode, robot, use_audio, instruction):
    """
    Command line interface for SAM2Act
    """

    cli_utils.setup_cli()

    episode_length = 25

    agent = Agent(model_path)
    agent.lang_goal = "push the buttons in the following order: red, green, blue."

    if instruction:
        agent.lang_goal = instruction

    if use_audio and robot.audio:
        with robot.audio as t:
            input("Speak and then press enter when done")
        x = t.process()
        logger.info("Transcribed %s", x)
        agent.lang_goal = x
    elif False:  # use this for prompting
        print("language goal. Press enter to accept or type a new one:")
        x = input(f"{agent.lang_goal}")
        if x:
            agent.lang_goal = x

    logger.info("Language goal is %s", agent.lang_goal)

    vis = SceneVisualizer(robot)
    vis.show()

    def process_action(action) -> None:
        action = np.asarray(action)
        action = CartesianAction.parse(action)

        logger.info("Action from agent %s", action)
        vis.update_action(action)

        if execution_mode == "auto":
            execute_action(robot, action)
        elif execution_mode == "step":
            prompt_for_action(robot, action)

    try:
        for i in range(episode_length):

            start_time = time.time()

            proprioception = robot.get_proprioception_data(True, True)
            perception = robot.get_vision_data()
            robot.update_wrist_camera_extrinsics(proprioception, perception)

            obs = {}
            obs.update(proprioception)
            obs.update(perception)

            logger.debug("getting scene")
            camera = robot.cameras[0]
            camera_name = camera.camera_name
            vis.update_scene(
                CameraData(
                    rgb_image=perception[f"{camera_name}_rgb"].transpose((2, 1, 0)),
                    depth_image=perception[f"{camera_name}_depth"],
                ),
                camera,
            )
            logger.debug("done getting scene")

            for camera in robot.cameras:
                obs[f"{camera.camera_name}_camera_extrinsics"] = np.linalg.inv(
                    obs[f"{camera.camera_name}_camera_extrinsics"]
                )

            observation = agent.prepare_observation(obs, i, episode_length)
            action = agent.get_action(None, observation)
            process_action(action.action)

            elapsed_time = time.time() - start_time
            logger.info("Processing action took %.2f seconds", elapsed_time)

            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        vis.close()

    logger.info("Done.")

    sys.exit(0)


if __name__ == "__main__":
    cli()

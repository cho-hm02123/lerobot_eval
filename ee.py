import os
import time
import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import (
    Robot,
    RobotConfig,
    so101_follower,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    so101_leader,
    make_teleoperator_from_config,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up, get_safe_torch_device
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

@dataclass
class HybridTeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    policy_path: str | None = None

def hybrid_teleop_loop(
    teleop: Teleoperator, robot: Robot, policy, fps: int, display_data: bool = False, duration: float | None = None
):
    # Determine the number of motor actions for terminal display
    display_len = max(len(key) for key in robot.action_features)
    start_time = time.perf_counter()
    policy_control_active = False

    while True:
        loop_start = time.perf_counter()
        
        # Get robot observation (state and camera feeds)
        observation = robot.get_observation()

        if not policy_control_active:
            # Teleoperation control
            action = teleop.get_action()
            
            # Gripper action detection logic (assuming gripper.pos from so101_leader)
            gripper_action_value = action.get("gripper.pos", 0)
            if gripper_action_value > 0.5:
                print("\n\n*** Gripper action detected. Switching to policy control. ***\n")
                policy_control_active = True
        else:
            # Policy control
            # Note: This requires a dataset object to get the features. 
            # This example assumes you have a way to define features for a policy without a dataset.
            # In a full script, you would get this from a dataset config.
            # For simplicity, we are assuming observation is already in the correct format.
            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )
            # The predict_action output might need to be converted to the format expected by `send_action`.
            # This is a simplification.
            print("\n\n*** Policy is controlling the robot. ***\n")

        sent_action = robot.send_action(action)
        
        if display_data:
            log_rerun_data(observation, sent_action)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        
        # Display current actions on terminal
        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in sent_action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {(time.perf_counter() - loop_start) * 1e3:.2f}ms ({1 / (time.perf_counter() - loop_start):.0f} Hz)")

        if duration is not None and time.perf_counter() - start_time >= duration:
            return

        move_cursor_up(len(sent_action) + 5)


@draccus.wrap()
def teleoperate_hybrid(cfg: HybridTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    
    # Initialize the policy using the provided path
    policy = SmolVLAPolicy.from_pretrained(cfg.policy_path)
    
    teleop.connect()
    robot.connect()

    try:
        hybrid_teleop_loop(teleop, robot, policy, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    # Example usage:
    # `teleoperate_hybrid` function should be called with appropriate configuration.
    # e.g., draccus.run(teleoperate_hybrid) in a command line context.
    # For this example, we directly call the loop logic with hardcoded values.
    # Note: `predict_action` and `build_dataset_frame` need a dataset object to define features.
    # This example demonstrates the logic, but a full working script would require
    # a proper dataset configuration or features object.
    pass


if __name__ == "__main__":
    main()

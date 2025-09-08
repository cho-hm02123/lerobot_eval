import os
import time
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.constants import HF_LEROBOT_HOME
import logging

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 10
FPS = 60
EPISODE_TIME_SEC = 100
TASK_DESCRIPTION = "Open the top blue drawer and Close the top blue drawer"
REPO_ID = "eval_open_and_close"
POLICY_REPO_ID = 'pretrained_model'
local_policy_path = os.path.join(os.path.expanduser("~/lerobotbi/lerobot/src/lerobot/scripts/outputs/train/open_and_close/checkpoints/last"), POLICY_REPO_ID)
local_dataset_path = os.path.join(os.path.expanduser("~/lerobotbi/lerobot/src/lerobot/nobana"), REPO_ID)

# Create the robot configuration
camera_config = {
    "grip": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    "top": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30),
    "depth": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30),
}
robot_config = SO101FollowerConfig(
    port="/dev/ttyFR",
    id="follwer_robot_arm",
    cameras=camera_config
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyTR",
    id="leader_robot_arm",
)

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

# Initialize the policy
policy = SmolVLAPolicy.from_pretrained(local_policy_path)
device = get_safe_torch_device(policy.config.device)
policy.to(device)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Initialize a flag for inference mode
events["start_inference"] = False

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# ----------------------------------------------------------------------------------------------------------------------

# **Phase 1: Teleoperation**
# This loop runs once for a single teleoperation episode.
print("Running teleoperation. Press SPACE to switch to inference mode.")

while not events["start_inference"]:
    action = teleop.get_action()
    robot.send_action(action)

# Check if the teleoperation episode should be saved

print("Teleoperation finished")
# ----------------------------------------------------------------------------------------------------------------------

# **Phase 2: Inference**
# This loop runs only if the spacebar was pressed during the teleoperation phase.
if events["start_inference"]:
    for episode_idx in range(NUM_EPISODES):
        print(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")
        
        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )
        dataset.save_episode()
        print(obs_features)
else:
    print("No inference requested. Exiting.")

# ----------------------------------------------------------------------------------------------------------------------

# Clean up
robot.disconnect()
teleop.disconnect()
# dataset.push_to_hub()

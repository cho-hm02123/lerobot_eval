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
# from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvia.modeling_smolvia import SmolVLAPolicy

from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.utils.robot_utils import busy_wait

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Pick up the red marker and put it in the box"
REPO_ID = "nobana/zzinmak"
local_dataset_path = os.path.join(os.path.expanduser("~/lerobotbi/lerobot/src/lerobot"), REPO_ID)

# Create the robot and teleoperator configurations
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

# Initialize the robot
robot = SO101Follower(robot_config)

# Initialize the policy
# policy = ACTPolicy.from_pretrained("<hf_username>/<my_policy_repo_id>")
policy = SmolVLAPolicy.from_pretrained("<hf_username>/<my_policy_repo_id>")
device = get_safe_torch_device(policy.config.device)
policy.to(device)

# Initialize the teleoperator
teleop = SO101Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Check for existing dataset and either create or resume
if os.path.exists(local_dataset_path):
    print("Found existing dataset. Resuming recording.")
    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=local_dataset_path,
    )
    dataset.start_image_writer(num_threads=4)
else:
    print("Creating a new dataset.")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=local_dataset_path,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Start the hybrid control loop for multiple episodes
for episode_idx in range(NUM_EPISODES):
    policy_control_active = False
    log_say(f"Starting hybrid control for episode {episode_idx + 1} of {NUM_EPISODES}. Using teleoperation.")

    start_episode_t = time.perf_counter()
    while not events["stop_recording"] and time.perf_counter() - start_episode_t < EPISODE_TIME_SEC:
        start_loop_t = time.perf_counter()

        observation = robot.get_observation()
        
        if not policy_control_active:
            # Teleoperation control
            action = teleop.get_action()

            user_choice = input("If you want to switch to policy inference, press Enter: ")
            if user_choice == '':
                print("Switching to policy control.")
                policy_control_active = True
        else:
            # Policy control
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_values = predict_action(
                observation_frame, policy, device, policy.config.use_amp, task=TASK_DESCRIPTION, robot_type=robot.robot_type
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        
        # Send action to the robot and record data
        sent_action = robot.send_action(action)

        action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        frame = {**observation_frame, **action_frame}
        dataset.add_frame(frame, task=TASK_DESCRIPTION)

        if events["display_data"]:
            log_rerun_data(observation, sent_action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / FPS - dt_s)

    # Save data after the episode ends
    if not events["stop_recording"]:
        log_say("Episode complete. Saving data...")
        dataset.save_episode()
    else:
        log_say("Stopping recording early. Discarding last episode...")
        dataset.stop_image_writer()
        dataset.clear_episode_buffer()
        dataset.start_image_writer(num_threads=4)
        break

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
# dataset.push_to_hub()
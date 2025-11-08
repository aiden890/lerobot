export PYTHONPATH=/Users/mingyeongho/Desktop/연구실/lerobot/src:$PYTHONPATH

python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AAF2186421 \
  --robot.cameras="{ up: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" \
  --robot.id=follower \
  --task="push the black object forward" \
  --server_address=115.145.175.11:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/khmin/lerobot/outputs/pi05_training/checkpoints/003000/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --debug_visualize_queue_size=True \
  --policy.use_vision_lora=0 \
  --policy.use_language_lora=0
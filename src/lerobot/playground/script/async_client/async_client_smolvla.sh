export PYTHONPATH=/Users/mingyeongho/Desktop/연구실/lerobot/src:$PYTHONPATH


python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AAF2641771 \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" \
  --robot.id=follower \
  --task="Grab the blue block and put in the cup" \
  --server_address=115.145.175.11:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=CSI-Agent/smolVLA_pnp144epi_8000 \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --debug_visualize_queue_size=True 
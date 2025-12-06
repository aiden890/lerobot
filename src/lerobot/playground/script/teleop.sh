lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AAF2186421 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fps: 30}}" \
    --robot.id=lerobot \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AAF2199371 \
    --teleop.id=blue \
    --display_data=true
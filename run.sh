source ~/.bashrc
ros2 launch lab_7.launch.py &
ros2 launch foxglove_bridge foxglove_bridge_launch.xml &
python hailo_detection.py

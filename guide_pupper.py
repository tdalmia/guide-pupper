# guide_pupper.py
# Group 2 Final Project for Stanford's CS 123: A Hands-On Introduction to Building AI-Enabled Robots
# Tushar Dalmia, Alex Gu, Neha Vinjapuri, Renee Duarte White, Cecelia Wu, Kevin Zhu

from enum import Enum
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import numpy as np
import time
import pygame
import os
import logging
import sounddevice as sd
from openai import OpenAI
import wave
import io
import threading

# vision processing constants
IMAGE_WIDTH = 1400
IMAGE_HEIGHT = 1400
CENTER_WIDTH_THRESHOLD = 0.08 # threshold for obstacle detected

# state machine constants
TIMEOUT = 3.0  # threshold in timer_callback
SEARCH_YAW_VEL = 5.0  # angular constant
TRACK_FORWARD_VEL = 0.2  # forward velocity constant
KP = 1.0  # proportional gain for tracking
STRIDE_TIME = 3.0 # forward duration after a turn
TRACK_TIME = 7.0 # forward duration after an obstacle

# constants to correct for slanted walking and turning
RIGHT_TURNING_TIME = 3.0
LEFT_TURNING_TIME = 2.5
ADJUST_CONSTANT = 0.013

# topics
VOICE_COMMAND_TOPIC = 'voice_command_topic'
USER_QUERY_TOPIC = 'user_query_topic'
OPENAI_API_KEY = ''  # removed for security

class State(Enum):
    WALK = 0
    TURN_RIGHT = 1
    RIGHT = 2
    READJUST_RIGHT = 3
    TRACK = 4
    TURN_LEFT = 5
    LEFT = 6
    READJUST_LEFT = 7
    STOP = 8
    # Removed SLOW_DOWN = 9

class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')

        # Initialize Logging
        self.setup_logging()

        # Publisher to send movement commands
        self.command_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Publisher for transcribed voice commands
        self.voice_command_publisher = self.create_publisher(
            String,
            VOICE_COMMAND_TOPIC,
            10
        )

        # Subscriber to vision data
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        # Subscriber to user voice commands (external commands)
        self.command_subscription = self.create_subscription(
            String,
            USER_QUERY_TOPIC,
            self.command_callback,
            10
        )

        # Subscriber to transcribed voice commands
        self.transcribed_command_subscription = self.create_subscription(
            String,
            VOICE_COMMAND_TOPIC,
            self.transcribed_command_callback,
            10
        )

        # Publisher for GPT-4 responses (optional)
        self.gpt4_publisher = self.create_publisher(
            String,
            GPT4_RESPONSE_TOPIC,
            10
        )

        # Initialize Pygame mixer once
        pygame.mixer.init()

        # Initialize state variables
        self.latest_time = time.time()
        self.center_x = 0
        self.center_box_ratio = 0
        self.center_width_ratio = 0
        self.state = State.WALK

        # Initialize speed multiplier (1.0 for normal speed, 0.5 for slow)
        self.speed_multiplier = 1.0  # **Added Speed Multiplier**

        # Define base durations for each state (at normal speed)
        self.base_durations = {
            State.TURN_RIGHT: RIGHT_TURNING_TIME,
            State.TURN_LEFT: LEFT_TURNING_TIME,
            State.RIGHT: STRIDE_TIME,
            State.LEFT: STRIDE_TIME,
            State.READJUST_RIGHT: LEFT_TURNING_TIME,
            State.READJUST_LEFT: RIGHT_TURNING_TIME,
            State.TRACK: TRACK_TIME
        }

        # Initialize timer for state machine
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Initialize variables for command handling
        self.current_command = None

        # Initialize OpenAI Whisper API client
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize a lock for thread-safe operations
        self.lock = threading.Lock()

        # Start audio processing in a separate thread
        self.audio_thread = threading.Thread(target=self.audio_recording_loop, daemon=True)
        self.audio_thread.start()

        self.get_logger().info('State Machine Node has started.')
        self.logger.info('State Machine Node has started.')

    def setup_logging(self):
        """
        Set up logging to file for transcriptions and command detections.
        """
        log_dir = "/home/pi/cs123pupper/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, "transcription_log.log")

        # Configure the logging
        self.logger = logging.getLogger('transcription_logger')
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)

    def detection_callback(self, msg):
        """
        Determine which of the HAILO detections is the most central detected object
        """
        normalized_x_vals = []
        normalized_y_vals = []
        object_area_ratio = []
        object_width_ratio = []

        for i, detection in enumerate(msg.detections):
            x_val = detection.bbox.center.position.x 
            y_val = detection.bbox.center.position.y

            box_width = detection.bbox.size_x
            box_height = detection.bbox.size_y

            normalized_x_val = (x_val - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
            normalized_x_vals.append(normalized_x_val)
            normalized_y_val = (y_val - IMAGE_HEIGHT / 2) / (IMAGE_HEIGHT / 2)
            normalized_y_vals.append(normalized_y_val)
            box_area_ratio = (box_width / IMAGE_WIDTH) * (box_height / IMAGE_HEIGHT)
            object_area_ratio.append(box_area_ratio)

            box_width_ratio = (box_width / IMAGE_WIDTH)
            object_width_ratio.append(box_width_ratio)

        if len(normalized_x_vals) == 0:
            self.center_x = 0
            self.center_box_ratio = 0
            self.center_width_ratio = 0
        else:
            min_index = np.argmin(np.abs(normalized_x_vals))
            self.center_x = normalized_x_vals[min_index]
            self.center_box_ratio = object_area_ratio[min_index]
            self.center_width_ratio = object_width_ratio[min_index]
            self.latest_time = time.time()

    def command_callback(self, msg):
        """
        Process voice commands received via ROS2 topic.
        Only handle 'stop' and 'slow_down' commands.
        """
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received external voice command: {command}")
        self.logger.info(f"Received external voice command: {command}")

        if command in ["stop", "pause"]:
            self.current_command = "stop"
            self.get_logger().info("Override command: STOP")
            self.logger.info("Override command: STOP")
            self.handle_stop()
        elif command in ["slow down", "slow"]:
            self.current_command = "slow_down"
            self.get_logger().info("Override command: SLOW_DOWN")
            self.logger.info("Override command: SLOW_DOWN")
            self.handle_slow_down()
        else:
            self.get_logger().warn(f"Ignored unknown command: {command}")
            self.logger.warning(f"Ignored unknown command: {command}")

    def transcribed_command_callback(self, msg):
        """
        Process transcribed voice commands received via 'voice_command_topic'.
        Only handle 'stop' and 'slow_down' commands.
        """
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received transcribed voice command: {command}")
        self.logger.info(f"Received transcribed voice command: {command}")

        if any(cmd in command for cmd in ["stop", "pause"]):
            self.handle_stop()
        elif any(cmd in command for cmd in ["slow down", "slow"]):
            self.handle_slow_down()
        else:
            self.get_logger().warn(f"Ignored unknown transcribed command: {command}")
            self.logger.warning(f"Ignored unknown transcribed command: {command}")

    def handle_stop(self):
        """
        Override current commands to stop the robot.
        """
        with self.lock:
            if self.state != State.STOP:
                self.state = State.STOP
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.command_publisher.publish(cmd)
                self.get_logger().info("Robot has been stopped.")
                self.logger.info("Robot has been stopped.")
                # Reset speed_multiplier to normal speed
                self.speed_multiplier = 1.0  # **Reset Speed Multiplier on Stop**

    def handle_slow_down(self):
        """
        Override current forward velocity to half speed.
        """
        with self.lock:
            if self.speed_multiplier != 0.75:
                self.speed_multiplier = 0.75  # reduce speed
                self.get_logger().info("Robot speed has been cut 25%.")
                self.logger.info("Robot speed has been cut 25%.")
            else:
                self.get_logger().info("Robot is already in slow mode.")
                self.logger.info("Robot is already in slow mode.")

    def timer_callback(self):
        """
        Implement a timer callback that sets the moves through the state machine based on if the time since the last detection is above a threshold TIMEOUT
        """
        yaw_command = 0.0
        forward_vel_command = 0.0

        # Handle STOP state by continuously publishing zero velocity
        with self.lock:
            if self.state == State.STOP:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.command_publisher.publish(cmd)
                self.get_logger().info("Robot is in STOP state. Velocity set to 0.")
                self.logger.info("Robot is in STOP state. Velocity set to 0.")
                return

            # Proceed with state transitions if not in STOP
            if self.state == State.WALK:
                # If center obstacle takes up more than threshold, turn right
                if self.center_width_ratio > CENTER_WIDTH_THRESHOLD:
                    self.state = State.TURN_RIGHT
                    self.time_started_turning_right = time.time()
                    self.get_logger().info('Object detected... Turning right...')
                    self.logger.info('Object detected... Turning right...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/object_detected.wav')
                    bark_sound.play()
            elif self.state == State.TURN_RIGHT:
                # If time since started turning right is greater than adjusted TURN_TURNING_TIME, go right
                adjusted_duration = self.base_durations[State.TURN_RIGHT] / self.speed_multiplier
                if time.time() - self.time_started_turning_right > adjusted_duration:
                    self.state = State.RIGHT
                    self.time_started_going_right = time.time()
                    self.get_logger().info('Walking straight...')
                    self.logger.info('Walking straight...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/walking_straight.wav')
                    bark_sound.play()
            elif self.state == State.RIGHT:
                # If time since started going right is greater than adjusted STRIDE_TIME, readjust
                adjusted_duration = self.base_durations[State.RIGHT] / self.speed_multiplier
                if time.time() - self.time_started_going_right > adjusted_duration:
                    self.state = State.READJUST_RIGHT
                    self.time_started_readjusting_right = time.time()
                    self.get_logger().info('Turning left...')
                    self.logger.info('Turning left...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/turning_left.wav')
                    bark_sound.play()
            elif self.state == State.READJUST_RIGHT:
                # If time since readjusting is greater than adjusted LEFT_TURNING_TIME, track
                adjusted_duration = self.base_durations[State.READJUST_RIGHT] / self.speed_multiplier
                if time.time() - self.time_started_readjusting_right > adjusted_duration:
                    self.state = State.TRACK 
                    self.time_started_tracking = time.time()
                    self.get_logger().info('Walking straight...')
                    self.logger.info('Walking straight...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/walking_straight.wav')
                    bark_sound.play()
            elif self.state == State.TRACK:
                # If time since tracking is greater than adjusted TRACK_TIME, turn left
                adjusted_duration = self.base_durations[State.TRACK] / self.speed_multiplier
                if time.time() - self.time_started_tracking > adjusted_duration: 
                    self.state = State.TURN_LEFT
                    self.time_started_turning_left = time.time()
                    self.get_logger().info('Turning left...')
                    self.logger.info('Turning left...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/turning_left.wav')
                    bark_sound.play()
            elif self.state == State.TURN_LEFT:
                # If time since started turning left is greater than adjusted TURN_LEFTING_TIME, go left
                adjusted_duration = self.base_durations[State.TURN_LEFT] / self.speed_multiplier
                if time.time() - self.time_started_turning_left > adjusted_duration:
                    self.state = State.LEFT
                    self.time_started_going_left = time.time()
                    self.get_logger().info('Walking straight...')
                    self.logger.info('Walking straight...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/walking_straight.wav')
                    bark_sound.play()
            elif self.state == State.LEFT:
                # If time since started going left is greater than adjusted STRIDE_TIME, readjust
                adjusted_duration = self.base_durations[State.LEFT] / self.speed_multiplier
                if time.time() - self.time_started_going_left > adjusted_duration:
                    self.state = State.READJUST_LEFT
                    self.time_started_readjusting_left = time.time()
                    self.get_logger().info('Turning right...')
                    self.logger.info('Turning right...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/turning_right.wav')
                    bark_sound.play()
            elif self.state == State.READJUST_LEFT:
                # If time since readjusting is greater than adjusted TURN_RIGHTING_TIME, go back to WALK
                adjusted_duration = self.base_durations[State.READJUST_LEFT] / self.speed_multiplier
                if time.time() - self.time_started_readjusting_left > adjusted_duration:
                    self.state = State.WALK
                    self.get_logger().info('Walking straight...')
                    self.logger.info('Walking straight...')
                    bark_sound = pygame.mixer.Sound('/home/pi/cs123pupper/pupper_llm/sounds/walking_straight.wav')
                    bark_sound.play()

            # State actions: Set movement commands based on current state
            if self.state == State.WALK:
                forward_vel_command = TRACK_FORWARD_VEL
                yaw_command = ADJUST_CONSTANT * SEARCH_YAW_VEL 
            elif self.state == State.TURN_RIGHT:
                yaw_command = -0.1 * SEARCH_YAW_VEL
            elif self.state == State.RIGHT:
                forward_vel_command = TRACK_FORWARD_VEL
                yaw_command = ADJUST_CONSTANT * SEARCH_YAW_VEL 
            elif self.state == State.READJUST_RIGHT:
                yaw_command = 0.1 * SEARCH_YAW_VEL
            elif self.state == State.TRACK:
                forward_vel_command = TRACK_FORWARD_VEL
                yaw_command = ADJUST_CONSTANT * SEARCH_YAW_VEL 
            elif self.state == State.TURN_LEFT:
                yaw_command = 0.1 * SEARCH_YAW_VEL
            elif self.state == State.LEFT:
                forward_vel_command = TRACK_FORWARD_VEL
                yaw_command = ADJUST_CONSTANT * SEARCH_YAW_VEL 
            elif self.state == State.READJUST_LEFT:
                yaw_command = -0.1 * SEARCH_YAW_VEL
            else:
                yaw_command = 0.0
                forward_vel_command = 0.0  # Default to no movement for other states

            # Apply speed multiplier
            forward_vel_command *= self.speed_multiplier
            yaw_command *= self.speed_multiplier

        # Publish movement commands
        cmd = Twist()
        cmd.angular.z = yaw_command
        cmd.linear.x = forward_vel_command
        self.command_publisher.publish(cmd)

    def publish_voice_command(self, message):
        """
        Publish the transcribed voice command to 'voice_command_topic'.
        """
        ros_msg = String()
        ros_msg.data = message
        self.voice_command_publisher.publish(ros_msg)
        self.get_logger().info(f"Published voice command: {message}")
        self.logger.info(f"Published voice command: {message}")

    def record_audio(self, duration=5, sample_rate=16000):
        """
        Record audio for a specified duration and return the audio data.
        """
        self.get_logger().info("Recording audio...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        self.get_logger().info("Audio recording finished.")
        return np.squeeze(audio_data)

    def audio_to_wav(self, audio_data, sample_rate=16000):
        """
        Convert numpy array audio to WAV format.
        """
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        wav_io.seek(0)
        return wav_io

    def transcribe_audio_with_whisper(self, filename):
        """
        Transcribe audio using Whisper API.
        """
        try:
            self.get_logger().info("Transcribing audio using Whisper API...")
            with open(filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            self.get_logger().info(f"Transcription response: {transcription}")
            self.logger.info(f"Transcription response: {transcription}")
            return transcription
        except Exception as e:
            self.get_logger().error(f"Error during transcription: {e}")
            self.logger.error(f"Error during transcription: {e}")
            return None

    def audio_recording_loop(self):
        """
        Continuously record and transcribe audio in a loop.
        """
        while rclpy.ok():
            try:
                self.get_logger().info("Audio loop: Starting recording...")
                audio_data = self.record_audio(duration=5.0)
                self.get_logger().info("Audio loop: Recording complete.")
                wav_io = self.audio_to_wav(audio_data)
                filename = '/home/pi/cs123pupper/audio_records/test_audio.wav'
                try:
                    with open(filename, 'wb') as f:
                        f.write(wav_io.read())
                    self.get_logger().info("Audio loop: Audio saved to test_audio.wav")
                    self.logger.info("Audio loop: Audio saved to test_audio.wav")
                except Exception as e:
                    self.get_logger().error(f"Audio loop: Failed to save audio file: {e}")
                    self.logger.error(f"Audio loop: Failed to save audio file: {e}")
                    continue

                # Transcribe audio using Whisper API
                t1 = time.time()
                self.get_logger().info("Audio loop: Starting transcription...")
                user_input = self.transcribe_audio_with_whisper(filename)
                t2 = time.time()
                self.get_logger().info(f"Audio loop: Time taken for transcription: {t2 - t1} seconds")
                self.logger.info(f"Audio loop: Time taken for transcription: {t2 - t1} seconds")

                # If the user said 'exit', stop the loop
                if user_input and user_input.lower() == 'exit':
                    self.get_logger().info("Audio loop: Received 'exit' command. Stopping audio recording loop.")
                    self.logger.info("Audio loop: Received 'exit' command. Stopping audio recording loop.")
                    break

                # Publish the recognized text
                if user_input:
                    self.get_logger().info(f"Audio loop: Publishing voice command: {user_input}")
                    self.publish_voice_command(user_input)

            except Exception as e:
                self.get_logger().error(f"Audio loop: Exception occurred: {e}")
                self.logger.error(f"Audio loop: Exception occurred: {e}")
                continue

            # Sleep for a short duration before next recording to prevent overlap
            time.sleep(0.5)

    def destroy_node(self):
        """
        Override destroy_node to ensure any resources are cleaned up gracefully.
        """
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    state_machine_node = StateMachineNode()

    try:
        rclpy.spin(state_machine_node)
    except KeyboardInterrupt:
        state_machine_node.get_logger().info("Program terminated by user")
        state_machine_node.logger.info("Program terminated by user")
    finally:
        # Stop the robot before shutting down
        zero_cmd = Twist()
        state_machine_node.command_publisher.publish(zero_cmd)
        state_machine_node.get_logger().info("Sent zero velocity command to stop the robot.")
        state_machine_node.logger.info("Sent zero velocity command to stop the robot.")

        state_machine_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

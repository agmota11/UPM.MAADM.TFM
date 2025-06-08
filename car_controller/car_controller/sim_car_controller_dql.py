#!/home/agmota/miniconda3/envs/ros2-tf/bin/python

from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import NavSatFix
import tensorflow as tf
import numpy as np
from collections import deque
import random
from pyproj import Geod
from car_controller.models import get_model_info
import datetime

class SimCarControllerDql(Node):
    def __init__(self):
        super().__init__('sim_car_controller_dql')

        # Model info
        model_info = get_model_info('transformer')
        json_file_path = model_info.json_path
        weights_file_path = model_info.weights_path

        self.get_logger().info(f'Loading {model_info.model_name} from {json_file_path} and weights from {weights_file_path}')
        sleep(3)
        self.model = self.load_model_from_json(json_file_path, weights_file_path)
        self.target_model = self.load_model_from_json(json_file_path, weights_file_path)

        self.counter = 20
        self.max_steering_angle = 0.4

        # DQL parameters
        self.gamma = 0.99
        self.epsilon = 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.train_freq = 5
        self.target_update_freq = 100
        self.train_step = 0

        # ROS2 subscriptions and publishers
        self.subscription_waypoints = self.create_subscription(
            Float32MultiArray, '/waypoints', self.waypoints_callback, 10)
        self.subscription_gps = self.create_subscription(
            NavSatFix, '/gps', self.gps_callback, 10)
        self.subscription_error = self.create_subscription(
            Float32, '/error', self.error_callback, 10)
        self.cmd_publisher = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.car_velocity = 5.0  # km/h
        self.waypoints = []
        self.latest_gps = None
        self.latest_error = 0.0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_next_state = None
        self.get_logger().info('SimCarControllerDql has been started.')

    def load_model_from_json(self, json_file_path, weights_file_path):
        with open(json_file_path, 'r') as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_file_path)

        # Freeze all layers except the last Dense
        for layer in model.layers:
            layer.trainable = False
        # Unfreeze the last Dense layer (change 'dense_1' if your last Dense has a different name)
        # Find the last Dense layer by type
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                layer.trainable = True
                break

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        self.get_logger().info("Model loaded, all layers frozen except the last Dense.")
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        self.get_logger().info("Target network updated.")

    def waypoints_callback(self, msg):
        self.waypoints = msg.data

    def gps_callback(self, msg):
        self.latest_gps = msg

    def error_callback(self, msg):
        self.latest_error = msg.data

    def build_ackermann_msg(self, velocity_in_km_s, steering_angle):
        velocity_meters_per_second = velocity_in_km_s / 3.6
        ack_msg = AckermannDriveStamped()
        ack_msg.header.frame_id = "map"
        ack_msg.drive.speed = velocity_meters_per_second
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.steering_angle_velocity = 0.0
        ack_msg.drive.acceleration = 0.0
        ack_msg.drive.jerk = 0.0
        return ack_msg

    def get_state(self):
        # Only use local waypoints as state (length 18)
        if not self.waypoints or self.latest_gps is None:
            return None
        waypoints_global = [(self.waypoints[i], self.waypoints[i + 1]) for i in range(0, len(self.waypoints), 2)]
        local_waypoints = calculate_local_coordinates(self.latest_gps.latitude, self.latest_gps.longitude, waypoints_global)
        local_waypoints_flattened = [coord for point in local_waypoints for coord in point][:18]
        # Pad with zeros if less than 18
        if len(local_waypoints_flattened) < 18:
            local_waypoints_flattened += [0.0] * (18 - len(local_waypoints_flattened))
        state = np.array(local_waypoints_flattened, dtype=np.float32)
        return state
    
    def select_action(self, state):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-self.max_steering_angle, self.max_steering_angle)
        q_values = self.model.predict(state.reshape(1, -1))
        return float(q_values[0][0])

    def timer_callback(self):
        state = self.get_state()
        if state is None:
            self.cmd_publisher.publish(self.build_ackermann_msg(0.0, 0.0))
            return

        if self.counter > 0:
            self.get_logger().warn('Initial steps')
            self.cmd_publisher.publish(self.build_ackermann_msg(self.car_velocity, 0.0))
            self.counter -= 1
            return
    
        if not self.waypoints:
            self.get_logger().warn('No waypoints available for prediction.')
            self.cmd_publisher.publish(self.build_ackermann_msg(0.0, 0.0))
            return

        # Select action
        action = self.select_action(state)
        action = np.clip(action, -self.max_steering_angle, self.max_steering_angle)

        # Publish action
        self.cmd_publisher.publish(self.build_ackermann_msg(self.car_velocity, action))

        # Store transition if not first step
        if self.last_state is not None:
            reward = np.clip(-abs(self.latest_error), -1.0, 0.0)  # Negative of the lateral error
            self.memory.append((self.last_state, self.last_action, reward, state, False))

        self.last_state = state
        self.last_action = action

        if len(self.memory) < self.batch_size * 15:
            return 

        # Train DQL
        if len(self.memory) >= self.batch_size and self.train_step % self.train_freq == 0:
            self.train_dql()
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_step += 1

    def train_dql(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Q-learning target
        target_q = self.model.predict(states)
        next_q = self.target_model.predict(next_states)
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_q[i])
            target_q[i][0] = target
        self.model.fit(states, target_q, epochs=1, verbose=0)

    def save_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/home/agmota/ros2_ws/UPM.MAADM.TFM/weights/dql_transformer_{timestamp}.h5"
        self.model.save_weights(save_path)
        self.get_logger().info(f"Model weights saved to {save_path}")

# Initialize Geod object for coordinate transformations
geod = Geod(ellps="WGS84")

def calculate_local_coordinates(lat, lon, waypoints):
    local_coords = []
    for wp_lat, wp_lon in waypoints:
        azimuth, _, distance = geod.inv(lon, lat, wp_lon, wp_lat)
        x = distance * np.cos(np.radians(azimuth))
        y = distance * np.sin(np.radians(azimuth))
        local_coords.append((x, y))
    return local_coords

def main(args=None):
    rclpy.init(args=args)
    node = SimCarControllerDql()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_model()  # Save model on exit
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
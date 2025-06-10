#!/home/agmota/miniconda3/envs/ros2-tf/bin/python

from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import NavSatFix
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from pyproj import Geod
import numpy as np
from car_controller.models import get_model_info

class SimCarController(Node):
    def __init__(self):
        super().__init__('sim_car_controller')

        model_key = 'transformer_dql3'
        model_info = get_model_info(model_key)
        json_file_path = model_info.json_path
        weights_file_path = model_info.weights_path
        
        self.get_logger().info(f'Loading {model_key} from {json_file_path} and weights from {weights_file_path}')
        sleep(3) # to watch log messages in the terminal
        self.model = self.load_model_from_json(json_file_path, weights_file_path)

        self.counter = 20
        self.max_steering_angle = 0.4

        # Subscriber to /waypoints topic
        self.subscription_waypoints = self.create_subscription(
            Float32MultiArray,
            '/waypoints',
            self.waypoints_callback,
            10
        )

        # Subscriber to /gps topic
        self.subscription_gps = self.create_subscription(
            NavSatFix,
            '/gps',
            self.gps_callback,
            10
        )

        # Publisher to /cmd topic
        self.cmd_publisher = self.create_publisher(AckermannDriveStamped, '/cmd', 10)

        # Timer to publish constant velocity and zero steering angle
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.car_velocity = 5.0  # km/h
        self.waypoints = []
        self.latest_gps = None
        self.get_logger().info('SimCarController has been started.')

    def load_model_from_json(self, json_file_path, weights_file_path):
        # Load JSON file
        with open(json_file_path, 'r') as json_file:
            model_json = json_file.read()

        # Create the model from the JSON configuration
        model = model_from_json(model_json)

        # Load weights into the model
        model.load_weights(weights_file_path)

        self.get_logger().info("Model loaded successfully from JSON and weights file.")
        return model

    def build_ackermann_msg(self, velocity_in_km_s, steering_angle):
        # Convert from km/h to m/s
        velocity_meters_per_second = velocity_in_km_s / 3.6

        ack_msg = AckermannDriveStamped()
        ack_msg.header.frame_id = "map"
        ack_msg.drive.speed = velocity_meters_per_second
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.steering_angle_velocity = 0.0  # [rad/s]
        ack_msg.drive.acceleration = 0.0  # [m/s^2]
        ack_msg.drive.jerk = 0.0  # [m/s^3]

        return ack_msg

    def waypoints_callback(self, msg):
        # Log the received waypoints
        waypoints = msg.data
        if not waypoints:
            self.get_logger().info('No waypoints received.')
        self.waypoints = waypoints

    def gps_callback(self, msg):
        # Update the latest GPS position
        self.latest_gps = msg

    def timer_callback(self):
        if not self.waypoints:
            self.get_logger().warn('No waypoints available for prediction.')
            self.cmd_publisher.publish(self.build_ackermann_msg(0.0, 0.0))
            return

        if self.latest_gps is None:
            self.get_logger().warn('No GPS data available.')
            self.cmd_publisher.publish(self.build_ackermann_msg(0.0, 0.0))
            return
        
        if self.counter > 0:
            self.get_logger().warn('Initial steps')
            self.cmd_publisher.publish(self.build_ackermann_msg(self.car_velocity, 0.0))
            self.counter -= 1
            return

        # Transform waypoints to local coordinates
        waypoints_global = [(self.waypoints[i], self.waypoints[i + 1]) for i in range(0, len(self.waypoints), 2)]
        local_waypoints = calculate_local_coordinates(self.latest_gps.latitude, self.latest_gps.longitude, waypoints_global)
        local_waypoints_flattened = [coord for point in local_waypoints for coord in point]

        # Trim excess waypoints if there are more than 9
        local_waypoints_flattened = local_waypoints_flattened[:18]
        input_data = tf.convert_to_tensor([local_waypoints_flattened], dtype=tf.float32)
        pred = self.model.predict(input_data)
        predicted_steering_angle = float(pred[0][0])
        predicted_steering_angle = max(min(predicted_steering_angle, self.max_steering_angle), -self.max_steering_angle)

        # Publish the predicted steering angle
        self.cmd_publisher.publish(self.build_ackermann_msg(self.car_velocity, predicted_steering_angle))
        self.get_logger().info(f'Published to /cmd: speed={self.car_velocity:.2f} km/h, steering_angle={predicted_steering_angle:.4f}')

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
    node = SimCarController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
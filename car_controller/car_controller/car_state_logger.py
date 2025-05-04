#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from eufs_msgs.msg import CarState
from sensor_msgs.msg import NavSatFix
from ackermann_msgs.msg import AckermannDriveStamped

import json
import os
from datetime import datetime

DATA_BASE_DIRECTORY = '/home/agmota/ros2_ws/UPM.MAADM.TFM/car_controller/data'

class CarStateLogger(Node):
    def __init__(self):
        super().__init__('car_state_logger')

        # QoS: Keep last 10 messages, reliable
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription_car = self.create_subscription(
            CarState,
            '/ground_truth/state',
            self.car_state_callback,
            qos
        )
        self.subscription_gps = self.create_subscription(
            NavSatFix,
            '/gps',
            self.gps_callback,
            qos
        )
        self.subscription_cmd = self.create_subscription(
            AckermannDriveStamped,
            '/cmd',
            self.cmd_callback,
            qos
        )

        self.latest_car_state = None
        self.latest_gps = None
        self.latest_cmd = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = os.path.join(DATA_BASE_DIRECTORY, f'car_log_{timestamp}.jsonl')
        self.log_buffer = []
        self.batch_size = 30
        self.get_logger().info(f'Logging to {self.output_file}')

    def car_state_callback(self, msg: CarState):
        self.latest_car_state = msg
        self.try_log()

    def gps_callback(self, msg: NavSatFix):
        self.latest_gps = msg
        self.try_log()

    def cmd_callback(self, msg: AckermannDriveStamped):
        self.latest_cmd = msg
        self.try_log()

    def try_log(self):
        if (self.latest_car_state is None 
            or self.latest_gps is None 
            or self.latest_cmd is None):
            return

        timestamp = self.extract_timestamp()
        speed = self.calculate_speed()
        yaw = self.calculate_yaw()
        steering_angle = self.get_steering_angle()
        latitude, longitude, altitude = self.extract_gps()

        log_entry = {
            "timestamp": timestamp,
            "speed_mps": speed,
            "yaw": yaw,
            "steering_angle": steering_angle,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude
        }

        self.get_logger().info(f"Logging: {log_entry}")

        self.log_buffer.append(log_entry)

        if len(self.log_buffer) >= self.batch_size:
            with open(self.output_file, "a") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry) + "\n")
            self.log_buffer = []

        self.latest_car_state = None
        self.latest_gps = None

    def extract_timestamp(self):
        330197000000
        stamp = self.latest_car_state.header.stamp
        return (stamp.sec * 10**9 + stamp.nanosec) // 1e6

    def calculate_speed(self):
        vx = self.latest_car_state.twist.twist.linear.x
        vy = self.latest_car_state.twist.twist.linear.y
        return (vx**2 + vy**2) ** 0.5

    def calculate_yaw(self):
        orientation = self.latest_car_state.pose.pose.orientation
        return 2 *  math.atan2(orientation.z, orientation.w)

    def extract_gps(self):
        gps = self.latest_gps
        return gps.latitude, gps.longitude, gps.altitude

    def get_steering_angle(self):
        return self.latest_cmd.drive.steering_angle
    
    def destroy_node(self):
        if self.log_buffer:
            with open(self.output_file, "a") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry) + "\n")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CarStateLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

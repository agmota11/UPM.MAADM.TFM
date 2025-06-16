#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from eufs_msgs.msg import CarState
from sensor_msgs.msg import NavSatFix
from insia_msg.msg import Telemetry, Telemetry2


import json
import os
from datetime import datetime

DATA_BASE_DIRECTORY = '/home/agmota/ros2_ws/UPM.MAADM.TFM/data'

class CarStateLogger(Node):
    def __init__(self):
        super().__init__('car_state_logger_insia')

        # QoS: Keep last 10 messages, reliable
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription_telemetry = self.create_subscription(
            Telemetry,
            '/telemetry',
            self.telemetry_callback,
            qos
        )
        self.subscription_telemetry2 = self.create_subscription(
            Telemetry2,
            '/telemetry2',
            self.telemetry2_callback,
            qos
        )

        self.latest_telemetry = None
        self.latest_telemetry2 = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = os.path.join(DATA_BASE_DIRECTORY, f'car_log_{timestamp}.jsonl')
        self.log_buffer = []
        self.batch_size = 30
        self.get_logger().info(f'Logging to {self.output_file}')

    def telemetry_callback(self, msg: Telemetry):
        self.latest_telemetry = msg
        self.try_log()

    def telemetry2_callback(self, msg: Telemetry2):
        self.latest_telemetry2 = msg
        self.try_log()

    def try_log(self):
        if (self.latest_telemetry is None 
            or self.latest_telemetry2 is None):
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

        self.latest_telemetry = None
        self.latest_telemetry2 = None

    def extract_timestamp(self):
        stamp = self.latest_telemetry.header.stamp
        return int(stamp.sec * 1e9 + stamp.nanosec) // 1_000_000  # ms

    def calculate_speed(self):
        # Use Telemetry speed (already in m/s)
        return self.latest_telemetry.speed

    def calculate_yaw(self):
        # Use Telemetry2 yaw (already in radians)
        return self.latest_telemetry2.yaw

    def extract_gps(self):
        # Use Telemetry2 latitude/longitude, altitude not available, set to 0.0
        return self.latest_telemetry2.latitude, self.latest_telemetry2.longitude, 0.0

    def get_steering_angle(self):
        # Use Telemetry steering (in radians)
        return self.latest_telemetry.steering

    
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

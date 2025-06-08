import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray, Float32
import csv
import matplotlib.pyplot as plt
from ackermann_msgs.msg import AckermannDriveStamped


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Load waypoints from the CSV file
        self.get_logger().info('Loading waypoints...')
        self.waypoints = self.load_waypoints('/home/agmota/ros2_ws/UPM.MAADM.TFM/data/datasets/waypoints_small_track.csv')
        self.current_waypoint_index = 0
        self.radius = 2.0  # Radius in meters to consider reaching a waypoint
        self.num_waypoints = 9
        self.num_laps = 10
        self.error = 0.0

        # QoS settings
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber to GPS data
        self.subscription_gps = self.create_subscription(
            NavSatFix,
            '/gps',
            self.gps_callback,
            qos
        )
                # Subscribe to /cmd for steering angle
        self.subscription_cmd = self.create_subscription(
            AckermannDriveStamped,
            '/cmd',
            self.cmd_callback,
            qos
        )
        self.latest_steering_angle = 0.0

        # Publisher to /waypoints
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, '/waypoints', 10)
        
        # Publisher to /error
        self.error_publisher = self.create_publisher(Float32, '/error', 10)

        # Timer to check car pose and publish waypoints
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.error_timer = self.create_timer(0.05, self.error_callback) # 20 Hz

        self.car_lat = None
        self.car_lon = None
        self.get_logger().info(f'WaypointNavigator has been started. {self.radius=}')

        # Initialize plot
        self.fig, self.ax = plt.subplots()

    def load_waypoints(self, file_path):
        waypoints = []
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    waypoint = [float(row['Latitude']), float(row['Longitude'])]
                    waypoints.append(waypoint)
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
        return waypoints

    def gps_callback(self, msg):
        self.car_lat = msg.latitude
        self.car_lon = msg.longitude

    
    def cmd_callback(self, msg):
        self.latest_steering_angle = msg.drive.steering_angle

    def timer_callback(self):
        if self.car_lon is None or self.car_lat is None:
            self.get_logger().warn('No GPS data received yet.')
            return

        # Get the current waypoint
        if self.current_waypoint_index >= len(self.waypoints) + 1:
            self.get_logger().info(f'All waypoints have been reached. {self.current_waypoint_index}/{len(self.waypoints) + 1}, Remaining laps: {self.num_laps}')
            self.waypoint_publisher.publish(Float32MultiArray())
            return

        current_index = self.current_waypoint_index % len(self.waypoints)
        current_waypoint = self.waypoints[current_index]
        next_waypoint = self.waypoints[self.current_waypoint_index + 1] if self.current_waypoint_index + 1 < len(self.waypoints) else self.waypoints[0]

        # Check if the car is within the radius of the current waypoint
        car_lat, car_lon = self.car_lat, self.car_lon
        waypoint_lat, waypoint_lon = current_waypoint[0], current_waypoint[1]
        next_waypoint_lat, next_waypoint_lon = next_waypoint[0], next_waypoint[1]

        distance = self.calculate_distance(car_lat, car_lon, waypoint_lat, waypoint_lon)
        next_distance = self.calculate_distance(car_lat, car_lon, next_waypoint_lat, next_waypoint_lon)

        if distance <= self.radius or next_distance <= distance + 1e-6:
            self.get_logger().info(f'Waypoint {self.current_waypoint_index} reached. Advancing to the next waypoint.')
            self.current_waypoint_index += 1
        
        if self.current_waypoint_index >= len(self.waypoints) + 1:
            self.current_waypoint_index = 0
            self.num_laps -= 1

        if self.num_laps == 0:
            self.get_logger().info(f'All waypoints have been reached. {self.current_waypoint_index}/{len(self.waypoints) + 1}, Remaining laps: {self.num_laps}')
            self.waypoint_publisher.publish(Float32MultiArray())
            return

        # Publish the next self.num_waypoints waypoints
        self.current_waypoints = self.waypoints[current_index:current_index + self.num_waypoints]
        if len(self.current_waypoints) < self.num_waypoints:
            remaining = self.num_waypoints - len(self.current_waypoints)
            self.current_waypoints += self.waypoints[1:remaining+1]
        
        path_msg = Float32MultiArray()
        path_msg.data = [coord for waypoint in self.current_waypoints for coord in waypoint]  # Flatten the list
        self.waypoint_publisher.publish(path_msg)
        self.get_logger().info(f'Published waypoints: {self.current_waypoint_index}/{len(self.waypoints) + 1}, Remaining laps: {self.num_laps}')

        # Plot the current position and waypoints
        self.plot_position_and_waypoints(car_lat, car_lon)

    def error_callback(self):
        if self.car_lat is None or self.car_lon is None:
            return
        
        # Calculate and publish lateral error for current position
        # Calculate lateral error
        lateral_error = self.calculate_lateral_error(self.car_lat, self.car_lon)
        
        current_index = self.current_waypoint_index % len(self.waypoints)
        current_waypoint = self.waypoints[current_index]

        dist_to_next_wp = self.calculate_distance(
            self.car_lat, self.car_lon,
            self.waypoints[(current_index + 1) % len(self.waypoints)][0],
            self.waypoints[(current_index + 1) % len(self.waypoints)][1]
        )

        # Add 2 times the distance to the next waypoint
        self.error = (lateral_error + 0.005 * dist_to_next_wp) / 100.0
        error_msg = Float32()
        error_msg.data = float(self.error)
        self.error_publisher.publish(error_msg)

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        # Haversine formula to calculate distance between two GPS coordinates
        R = 6371000  # Radius of Earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def calculate_lateral_error(self, car_lat, car_lon):
        # Convert all waypoints to (x, y) in meters using the first waypoint as reference
        ref_lat, ref_lon = self.waypoints[0]
        car_xy = self.lonlat_to_xy(car_lon, car_lat, ref_lon, ref_lat)
        waypoints_xy = [self.lonlat_to_xy(wp[1], wp[0], ref_lon, ref_lat) for wp in self.waypoints]

        # Find the minimum distance from the car to any segment between waypoints
        min_dist = float('inf')
        for i in range(len(waypoints_xy) - 1):
            seg_start = waypoints_xy[i]
            seg_end = waypoints_xy[i+1]
            dist = self.point_to_segment_distance(car_xy, seg_start, seg_end)
            min_dist = min(min_dist, dist)
        return min_dist

    def lonlat_to_xy(self, lon, lat, ref_lon, ref_lat):
        # Equirectangular projection
        R = 6371000  # Earth radius in meters
        x = (math.radians(lon) - math.radians(ref_lon)) * R * math.cos(math.radians(ref_lat))
        y = (math.radians(lat) - math.radians(ref_lat)) * R
        return (x, y)

    def point_to_segment_distance(self, pt, seg_start, seg_end):
        # pt, seg_start, seg_end: (x, y)
        v = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        w = (pt[0] - seg_start[0], pt[1] - seg_start[1])
        c1 = v[0]*w[0] + v[1]*w[1]
        c2 = v[0]*v[0] + v[1]*v[1]
        b = c1 / c2 if c2 != 0 else 0
        b = max(0, min(1, b))
        pb = (seg_start[0] + b * v[0], seg_start[1] + b * v[1])
        return math.hypot(pt[0] - pb[0], pt[1] - pb[1])

    def plot_position_and_waypoints(self, car_lat, car_lon):
        # Clear the previous plot
        self.ax.clear()

        # Plot the next self.num_waypoints waypoints
        waypoints_to_plot = self.waypoints[self.current_waypoint_index:self.current_waypoint_index + self.num_waypoints]
        if len(waypoints_to_plot) < self.num_waypoints:
            remaining = self.num_waypoints - len(waypoints_to_plot)
            waypoints_to_plot += self.waypoints[1:remaining+1]

        waypoint_lats = [wp[0] for wp in waypoints_to_plot]
        waypoint_lons = [wp[1] for wp in waypoints_to_plot]
        self.ax.plot(waypoint_lons, waypoint_lats, 'bo-', label='Next Waypoints')

        # Plot the car's current position
        self.ax.plot(car_lon, car_lat, 'ro', label='Current Position')

        # Plot all waypoints with 0.1 alpha
        all_waypoint_lats = [wp[0] for wp in self.waypoints]
        all_waypoint_lons = [wp[1] for wp in self.waypoints]
        self.ax.plot(all_waypoint_lons, all_waypoint_lats, 'go-', alpha=0.1, label='All Waypoints')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.legend()

        # Annotate lateral error and steering angle in the center of the plot
        self.ax.text(0.5, 0.55, f"Steering Angle: {self.latest_steering_angle:.3f}",
                 transform=self.ax.transAxes, fontsize=10, verticalalignment='center',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        self.ax.text(0.5, 0.45, f"Error: {self.error:.3f} m",
                 transform=self.ax.transAxes, fontsize=10, verticalalignment='center',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        
        # Update the plot
        self.fig.canvas.draw()
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.waypoint_publisher.publish(Float32MultiArray())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray
import csv
import matplotlib.pyplot as plt

class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Load waypoints from the CSV file
        self.get_logger().info('Loading waypoints...')
        self.waypoints = self.load_waypoints('/home/agmota/ros2_ws/UPM.MAADM.TFM/data/datasets/waypoints_small_track.csv')
        self.current_waypoint_index = 0
        self.radius = 2.0  # Radius in meters to consider reaching a waypoint
        self.num_waypoints = 9

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

        # Publisher to /waypoints
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, '/waypoints', 10)

        # Timer to check car pose and publish waypoints
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.latest_gps = None
        self.get_logger().info('WaypointNavigator has been started.')

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
        self.latest_gps = msg

    def timer_callback(self):
        if self.latest_gps is None:
            self.get_logger().warn('No GPS data received yet.')
            return

        # Get the current waypoint
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('All waypoints have been reached.')
            return

        current_waypoint = self.waypoints[self.current_waypoint_index]

        # Check if the car is within the radius of the current waypoint
        car_lat = self.latest_gps.latitude
        car_lon = self.latest_gps.longitude
        waypoint_lat = current_waypoint[0]
        waypoint_lon = current_waypoint[1]

        distance = self.calculate_distance(car_lat, car_lon, waypoint_lat, waypoint_lon)

        if distance <= self.radius:
            self.get_logger().info(f'Waypoint {self.current_waypoint_index} reached. Advancing to the next waypoint.')
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.get_logger().info('All waypoints have been reached.')
                return

        # Publish the next self.num_waypoints waypoints
        self.current_waypoints = self.waypoints[self.current_waypoint_index:self.current_waypoint_index + self.num_waypoints]
        if len(self.current_waypoints) < self.num_waypoints:
            remaining = self.num_waypoints - len(self.current_waypoints)
            self.current_waypoints += self.waypoints[1:remaining+1]

        if self.current_waypoint_index > len(self.waypoints):
            self.get_logger().info('No waypoints to publish.')
            self.waypoint_publisher.publish(Float32MultiArray())
            return
        
        path_msg = Float32MultiArray()
        path_msg.data = [coord for waypoint in self.current_waypoints for coord in waypoint]  # Flatten the list
        self.waypoint_publisher.publish(path_msg)
        self.get_logger().info(f'Published waypoints: {len(self.current_waypoints)}')

        # Plot the current position and waypoints
        self.plot_position_and_waypoints(car_lat, car_lon)

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

        # Add labels and legend
        # Plot all waypoints with 0.1 alpha
        all_waypoint_lats = [wp[0] for wp in self.waypoints]
        all_waypoint_lons = [wp[1] for wp in self.waypoints]
        self.ax.plot(all_waypoint_lons, all_waypoint_lats, 'go-', alpha=0.1, label='All Waypoints')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.legend()

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
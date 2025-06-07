from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='car_controller',
            executable='sim_car_controller',
            name='sim_car_controller'
        ),
        Node(
            package='car_controller',
            executable='car_state_logger',
            name='car_state_logger'
        ),
        Node(
            package='car_controller',
            executable='waypoint_controller',
            name='waypoint_controller'
        ),
    ])
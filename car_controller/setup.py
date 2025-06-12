from setuptools import setup

package_name = 'car_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agmota',
    maintainer_email='a.gmota@alumnos.upm.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_state_logger = car_controller.car_state_logger:main',
            'waypoint_controller = car_controller.waypoint_controller:main',
            'sim_car_controller = car_controller.sim_car_controller:main',
            'sim_car_controller_rf = car_controller.sim_car_controller_rf:main',
            'sim_car_controller_xgb = car_controller.sim_car_controller_xgb:main',
            'sim_car_controller_dql = car_controller.sim_car_controller_dql:main',
            'sim_car_controller_dql_alt = car_controller.sim_car_controller_dql_alt:main',
        ],
    },
)

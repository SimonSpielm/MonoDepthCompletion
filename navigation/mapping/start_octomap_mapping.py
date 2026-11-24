#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            emulate_tty=True, #helps for printing

            parameters=[{
                # --- Auflösung & Frames ---
                "resolution": 0.02,
                #"world_frame_id": "map",
                "base_frame_id": "ground",       # only for ground-filter
                "frame_id": "map",

                # --- Sensor-Modell ---
                "sensor_model.hit": 0.85,
                "sensor_model.miss": 0.25,
                "sensor_model.min": 0.12,
                "sensor_model.max": 0.97,
                "sensor_model.max_range": 5.0, 

                # --- Max/Min Height from Wt
                "point_cloud_min_z": -3.0,
                "point_cloud_max_z":  3.0,

                # --- Filter / Qualität ---
                "filter_speckles": True,
                "filter_ground_plane": False,     # ground filter aus = weniger Rechenaufwand
                "ground_filter.distance": 0.1,
                "ground_filter.angle":    0.1,
                "ground_filter.plane_distance": 0.05,

                # --- Map-Speicher / Kompression ---
                "compress_map": False,       # schneller, braucht mehr RAM; True = kleiner, langsamer

                # --- 2D-Map & weitere Outputs ---
                "publish_2d_map": False,
                "incremental_2D_projection": False,
                "project_complete_map": False,
                "publish_free_space": False,
                "incremental_2D_projection": False,
                
                # --- Latching / Re-publish Verhalten ---
                "latch": True,

                "use_height_map": False,
                "use_colored_map": False,
                'use_sim_time': True,
            }],

            remappings=[
                ('cloud_in', '/camera/filtered/points')   # pointcloud_publisher.py,
            ],

            # Log-Level runterdrehen, wenn alles tut:
            arguments=['--ros-args', '--log-level', 'octomap_server:=debug'],
        )
    ])
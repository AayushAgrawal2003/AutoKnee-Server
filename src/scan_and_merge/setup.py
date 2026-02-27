from setuptools import setup
import os
from glob import glob

package_name = "scan_and_merge"

setup(
    name=package_name,
    version="0.1.0",
    packages=["scan_and_merge"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "resource"), glob("resource/*.ply")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aayush",
    maintainer_email="todo@todo.com",
    description="Scan and merge point clouds using KUKA Med 7 + RealSense",
    license="MIT",
    entry_points={
        "console_scripts": [
            "scan_and_merge_node = scan_and_merge.scan_and_merge_node:main",
            "detect_and_merge_node = scan_and_merge.detect_and_merge_node:main",
            "replay_waypoints = scan_and_merge.replay_waypoints:main",
            "replay_trajectory = scan_and_merge.replay_trajectory:main",
            "cloud_publisher = scan_and_merge.cloud_publisher:main",
        ],
    },
)
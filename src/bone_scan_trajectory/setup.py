from setuptools import find_packages, setup

package_name = "bone_scan_trajectory"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "bone_scan_trajectory_node = bone_scan_trajectory.bone_scan_trajectory_node:main",
        ],
    },
)

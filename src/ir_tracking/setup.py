from setuptools import setup
from glob import glob
import os

package_name = "ir_tracking"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "roms"), glob("roms/*.rom")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aayush",
    maintainer_email="todo@todo.com",
    description="NDI Polaris Vega IR bone tracker integration for KUKA surgical navigation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ir_tracking_node = ir_tracking.ir_tracking_node:main",
        ],
    },
)

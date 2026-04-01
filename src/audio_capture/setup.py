from setuptools import setup

package_name = "audio_capture"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aayush",
    maintainer_email="todo@todo.com",
    description="Publish microphone audio to a ROS 2 topic",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mic_publisher = audio_capture.mic_publisher:main",
            "speech_recognizer = audio_capture.speech_recognizer:main",
        ],
    },
)

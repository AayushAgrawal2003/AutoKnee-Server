import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension
import sounddevice as sd


class MicPublisher(Node):
    def __init__(self):
        super().__init__('mic_publisher')

        self.declare_parameter('device_name', 'CMTECK')
        self.declare_parameter('sample_rate', 44100)
        self.declare_parameter('channels', 2)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('topic', '/audio')

        device_name = self.get_parameter('device_name').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.chunk_size = self.get_parameter('chunk_size').value
        topic = self.get_parameter('topic').value

        self.publisher_ = self.create_publisher(Int16MultiArray, topic, 10)

        # Find the CMTECK device index
        self.device_index = None
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if device_name.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                self.device_index = i
                self.get_logger().info(f"Found device: {d['name']} (index {i})")
                break

        if self.device_index is None:
            self.get_logger().error(f"No input device matching '{device_name}' found!")
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    self.get_logger().info(f"  [{i}] {d['name']}")
            raise RuntimeError(f"Device '{device_name}' not found")

        self.get_logger().info(
            f"Publishing audio on '{topic}' — "
            f"rate={self.sample_rate}, channels={self.channels}, chunk={self.chunk_size}"
        )

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"Audio status: {status}")

        msg = Int16MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='frames', size=frames, stride=frames * self.channels),
            MultiArrayDimension(label='channels', size=self.channels, stride=self.channels),
        ]
        msg.data = indata.flatten().tolist()
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.stream.stop()
        self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MicPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

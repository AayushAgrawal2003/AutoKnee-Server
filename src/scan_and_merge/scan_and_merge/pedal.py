import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import serial

class PedalNode(Node):
    def __init__(self):
        super().__init__('foot_pedal_node')
        self.publisher_ = self.create_publisher(Bool, '/pedal_press', 10)
        self.serial_ = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        self.timer = self.create_timer(0.01, self.read_serial)

    def read_serial(self):
        if self.serial_.in_waiting > 0:
            line = self.serial_.readline().decode('utf-8').strip()
            if line in ('0', '1'):
                msg = Bool()
                msg.data = line == '1'
                self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PedalNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped


class EEPosePublisher(Node):
    def __init__(self):
        super().__init__('marker_pub')

        # --- Parameters ---
        self.declare_parameter('base_frame', 'link_0')
        self.declare_parameter('ee_frame', 'ee_marker_center')
        self.declare_parameter('rate', 50.0)  # Hz

        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        rate = self.get_parameter('rate').value

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Publisher ---
        self.pub = self.create_publisher(PoseStamped, '/ee_marker_center', 10)

        # --- Timer ---
        self.timer = self.create_timer(1.0 / rate, self.publish_pose)
        # self.get_logger().info(
        #     f'Publishing EE pose: {self.base_frame} → {self.ee_frame} at {rate} Hz')

    def publish_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time())

            msg = PoseStamped()
            msg.header = t.header
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation

            self.pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = EEPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
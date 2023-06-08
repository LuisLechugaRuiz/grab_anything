import rclpy
from rclpy_msgs.msg import String
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class SegmentateImageNode(Node):
    def __init__(self):
        super().__init__("image_saver")

        self.bridge = CvBridge()
        self.last_image = None

        self.image_subscriber = self.create_subscription(
            Image, "camera_topic", self.image_callback, 10
        )

        # TODO: Change by whisper + ChatGPT activation
        self.save_command_subscriber = self.create_subscription(
            String, "save_image_command", self.save_image_callback, 10
        )

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def save_image_callback(self, msg):
        if self.last_image is not None:
            filename = msg.data  # assuming the message contains the filename
            cv2.imwrite(filename, self.last_image)
            self.get_logger().info("Image saved to " + filename)
        else:
            self.get_logger().info("No image data received yet.")
    
    def segmentate_image(self, image_path):
        

def main(args=None):
    rclpy.init(args=args)

    segmentate_image = SegmentateImageNode()

    rclpy.spin(segmentate_image)

    segmentate_image.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

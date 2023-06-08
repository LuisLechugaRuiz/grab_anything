import rclpy
from rclpy_msgs.msg import String
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit.planning import MoveItPy
from shape_msgs.msg import Mesh
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import numpy as np

from grab_anything.libs.segmentate_image import SegmentateImage


# Ideally it should be splitted into 3 nodes:
# - Perception.
# - Decision making.
# - Motion planning. (MoveIt)
class GrabObjectNode(Node):
    def __init__(self):
        super().__init__("image_saver")

        self.bridge = CvBridge()
        self.last_image = None

        self.image_subscriber = self.create_subscription(
            Image, "camera_topic", self.image_callback, 10
        )
        self.depth_image_subscriber = self.create_subscription(
            Image, "depth_camera_topic", self.depth_image_callback, 10
        )

        # TODO: Change by whisper + ChatGPT activation
        self.process_image_subscriber = self.create_subscription(
            String, "process_image", self.process_image_callback, 10
        )
        self.image_path = "last_image.png"
        self.segmentate_image = SegmentateImage()

        self.panda = MoveItPy(node_name="moveit_py_planning_scene")
        self.panda_arm = self.panda.get_planning_component("panda_arm")
        self.panda_arm.set_start_state(configuration_name="ready")
        self.panda_arm.set_goal_state(configuration_name="extended")
        self.planning_scene_monitor = self.panda.get_planning_scene_monitor()

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_image_callback(self, msg):
        self.last_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def process_image_callback(self, msg):
        if self.last_image is not None:
            cv2.imwrite(self.image_path, self.last_image)
            self.get_logger().info("Image saved to " + self.image_path)
            masks, _, labels = self.segmentate_image.get_objects(self.image_path)
            object = msg.data
            # We shoud get an object id from the object as this is just a label.
            if object in labels:
                points_3d = self.get_3d_points(
                    mask=masks[labels.index(msg.data)],
                    depth_image=self.last_depth_image,
                )
                self.get_logger().info("Object found!")
                self.add_to_planning_scene(points_3d)
                self.move_to_object(object_id=object)
                # Ideal flow:
                # Find grasping point for the segmented object.
                # Move closer to the grasping point.
                # Open gripper.
                # Move to grasping point.
                # Close gripper.
                # Go to dispose point.
        else:
            self.get_logger().info("No image data received yet.")

    def get_3d_points(self, mask, depth_image):
        # Get the x,y coordinates of all non-zero values in the mask
        coords = np.nonzero(mask)

        # Find the corresponding depth values
        depth_values = depth_image[coords]

        # Create the 3D points (x,y are from the image, z is from the depth)
        points_3d = np.column_stack((coords[1], coords[0], depth_values))

        return points_3d

    def add_to_planning_scene(self, points_3d, object_id="object"):
        # Convert points to PointCloud2
        ros_point_cloud = pc2.create_cloud_xyz32(
            header=Header(frame_id="panda_link0"), points=points_3d
        )

        with self.planning_scene_monitor.read_write() as scene:
            # Create a collision object
            collision_object = CollisionObject()
            collision_object.id = object_id
            collision_object.header.frame_id = "panda_link0"
            # Assume object is a mesh for now
            collision_object.meshes = [Mesh()]
            collision_object.meshes[0].triangles = ros_point_cloud
            collision_object.meshes[0].vertices = points_3d
            collision_object.operation = CollisionObject.ADD

            # Add the object to the planning scene
            scene.apply_collision_object(collision_object)
            scene.current_state.update()  # Important to ensure the scene is updated

    def move_to_object(self, object_id="object"):
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions(
                "panda_arm"
            )

            # Find the min and max 3D points for the segmented object
            object_points = np.array(
                [
                    point
                    for point in scene.world.collision_objects[object_id]
                    .primitives[0]
                    .vertices
                ]
            )
            min_point = np.min(object_points, axis=0)
            max_point = np.max(object_points, axis=0)

            # Calculate the mean of the min and max points to find the center
            center_point = (min_point + max_point) / 2

            # Set the pose goal as the center of the segmented object
            pose_goal = Pose()
            pose_goal.position.x = center_point[0]
            pose_goal.position.y = center_point[1]
            pose_goal.position.z = center_point[2]
            pose_goal.orientation.w = 1.0

            # Set the robot state and check collisions
            robot_state.set_from_ik("panda_arm", pose_goal, "panda_hand")
            robot_state.update()  # required to update transforms
            robot_collision_status = scene.is_state_colliding(
                robot_state=robot_state,
                joint_model_group_name="panda_arm",
                verbose=True,
            )
            self.get_logger().info(
                f"\nRobot is in collision: {robot_collision_status}\n"
            )

            # Collide with the object just to show the motion planning, but need improvements.
            self.panda_arm.go_to_state(robot_state, wait=True)

            # Restore the original state
            robot_state.set_joint_group_positions("panda_arm", original_joint_positions)
            robot_state.update()  # required to update transforms


def main(args=None):
    rclpy.init(args=args)

    grab_object_node = GrabObjectNode()

    rclpy.spin(grab_object_node)

    grab_object_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


class PointCloudDenseBridge:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/Alpha/velodyne_points")
        self.output_topic = rospy.get_param("~output_topic", "/Alpha/velodyne_points_dense")
        self.log_every = int(rospy.get_param("~log_every", 100))
        self.msg_count = 0

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=2)
        self.sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.callback,
            queue_size=2,
            buff_size=2 ** 24,
        )

        rospy.loginfo(
            "[pointcloud_dense_bridge] %s -> %s",
            self.input_topic,
            self.output_topic,
        )

    def callback(self, msg: PointCloud2):
        self.msg_count += 1
        field_names = [f.name for f in msg.fields]

        points = list(point_cloud2.read_points(
            msg,
            field_names=field_names,
            skip_nans=True,
        ))
        if not points:
            return

        out = point_cloud2.create_cloud(msg.header, msg.fields, points)
        out.is_dense = True
        out.height = 1
        out.width = len(points)
        out.header = msg.header
        self.pub.publish(out)

        if self.msg_count % self.log_every == 0:
            total_in = msg.width * msg.height
            rospy.loginfo(
                "[pointcloud_dense_bridge] msg=%d in=%d out=%d dropped=%d",
                self.msg_count,
                total_in,
                len(points),
                total_in - len(points),
            )


def main():
    rospy.init_node("pointcloud_dense_bridge")
    PointCloudDenseBridge()
    rospy.spin()


if __name__ == "__main__":
    main()

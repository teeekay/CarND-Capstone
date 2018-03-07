#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from styx_msgs.msg import Lane, TrafficLightArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA

'''
This node publishes a visualization_marker_array topic for RViz to do 3D
visualization of the car driving around the waypoints and shows the ground
truth traffic light colors to compare with the detected status.
'''


class VisNode(object):

    def __init__(self):
        rospy.init_node('vis_node')

        self.vis_enabled = rospy.get_param('~vis_enabled', False)
        self.vis_rate = rospy.get_param('~vis_rate', 5)

        self.subs = {}
        self.pubs = {}
        self.current_pose = None
        self.waypoints = []
        self.traffic_lights = []

        self.pubs['/visualization_marker_array'] = rospy.Publisher(
                    '/visualization_marker_array', MarkerArray, queue_size=1)

        self.subs['/current_pose'] = rospy.Subscriber(
                    '/current_pose', PoseStamped, self.handle_current_pose_msg)

        self.subs['/base_waypoints'] = rospy.Subscriber(
                    '/base_waypoints', Lane, self.handle_waypoints_msg)

        self.subs['/vehicle/traffic_lights'] = rospy.Subscriber(
                    '/vehicle/traffic_lights', TrafficLightArray,
                    self.handle_traffic_lights_msg)

        self.loop()

    # Callback to handle ground truth traffic lights message
    def handle_current_pose_msg(self, current_pose_msg):
        self.current_pose = current_pose_msg.pose

    # Callback to handle base waypoints message
    def handle_waypoints_msg(self, lane_msg):
        for waypoint in lane_msg.waypoints:
            self.waypoints.append(waypoint)
        self.subs['/base_waypoints'].unregister()

    # Callback to handle ground truth traffic lights message
    def handle_traffic_lights_msg(self, traffic_lights_msg):
        self.traffic_lights = []
        for traffic_light in traffic_lights_msg.lights:
            self.traffic_lights.append(traffic_light)

    # Helper function to set traffic light RGBA color from status #
    def set_traffic_light_color(self, tl_status, alpha):
        tl_color = ColorRGBA()
        tl_color.a = alpha
        if tl_status == 0:
            # Red light
            tl_color.r = 1.0
            tl_color.g = 0.0
            tl_color.b = 0.0
        elif tl_status == 1:
            # Yellow light
            tl_color.r = 1.0
            tl_color.g = 1.0
            tl_color.b = 0.0
        elif tl_status == 2:
            # Green light
            tl_color.r = 0.0
            tl_color.g = 1.0
            tl_color.b = 0.0
        else:
            # White light (unknown status)
            tl_color.r = 1.0
            tl_color.g = 1.0
            tl_color.b = 1.0
        return tl_color

    def loop(self):
        rate = rospy.Rate(self.vis_rate)
        while not rospy.is_shutdown():
            if self.vis_enabled:
                marker_array = MarkerArray()

                # Marker[0] = Car Pose Arrow
                if self.current_pose is not None:
                    car_marker = Marker()
                    car_marker.id = 0
                    car_marker.header.frame_id = "/world"
                    car_marker.header.stamp = rospy.Time()
                    car_marker.type = Marker.ARROW
                    car_marker.action = Marker.ADD
                    car_marker.scale.x = 40.0
                    car_marker.scale.y = 8.0
                    car_marker.scale.z = 8.0
                    car_marker.color.a = 1.0
                    car_marker.color.r = 1.0
                    car_marker.color.g = 0.0
                    car_marker.color.b = 1.0
                    car_marker.pose = self.current_pose
                    marker_array.markers.append(car_marker)

                # Marker[1] = Base Waypoint Line
                base_wp_line = Marker()
                base_wp_line.id = 1
                base_wp_line.header.frame_id = "/world"
                base_wp_line.header.stamp = rospy.Time()
                base_wp_line.type = Marker.LINE_STRIP
                base_wp_line.action = Marker.ADD
                base_wp_line.scale.x = 3.0
                base_wp_line.scale.y = 3.0
                base_wp_line.scale.z = 3.0
                base_wp_line.color.a = 1.0
                base_wp_line.color.r = 1.0
                base_wp_line.color.g = 1.0
                base_wp_line.color.b = 1.0
                base_wp_line.pose.orientation.w = 1.0
                for wp_idx in range(len(self.waypoints)):
                    if wp_idx % 30 == 0:
                        base_wp_line.points.append(
                            self.waypoints[wp_idx].pose.pose.position)
                marker_array.markers.append(base_wp_line)

                # Marker[2] = Traffic Waypoint Spheres
                tl_marker = Marker()
                tl_marker.id = 2
                tl_marker.header.frame_id = "/world"
                tl_marker.header.stamp = rospy.Time()
                tl_marker.type = Marker.SPHERE_LIST
                tl_marker.action = Marker.ADD
                tl_marker.scale.x = 30.0
                tl_marker.scale.y = 30.0
                tl_marker.scale.z = 30.0
                for tl_idx in range(len(self.traffic_lights)):
                    tl_marker.points.append(
                                self.traffic_lights[tl_idx].pose.pose.position)

                    tl_color = self.set_traffic_light_color(
                                        self.traffic_lights[tl_idx].state, 0.6)

                    tl_marker.colors.append(tl_color)
                marker_array.markers.append(tl_marker)

                # Marker[3] = Detected Traffic Waypoint Cubes
                det_tl_marker = Marker()
                det_tl_marker.id = 3
                det_tl_marker.header.frame_id = "/world"
                det_tl_marker.header.stamp = rospy.Time()
                det_tl_marker.type = Marker.CUBE_LIST
                det_tl_marker.action = Marker.ADD
                det_tl_marker.scale.x = 30.0
                det_tl_marker.scale.y = 30.0
                det_tl_marker.scale.z = 30.0
                # TODO - Add after traffic light detection is implemented

                self.pubs['/visualization_marker_array'].publish(marker_array)

            rate.sleep()


if __name__ == '__main__':
    VisNode()

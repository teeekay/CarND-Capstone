#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        # Use Ground truth to stop the car till traffic light classifier is implemented
        self.use_ground_truth = rospy.get_param("~use_tl_ground_truth",True)
        # to stop using gt, do this -> rospy.set_param('~use_tl_ground_truth', False)




        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            waypoints : set of points to look for 

        Returns:
            int: index of the closest waypoint in waypoints array

        """
        min_dist = float("inf")
        closest_wp_idx = -1

        for idx, wp in enumerate(waypoints):
            dist = self.find_dist(pose, wp.pose.pose)
            if(dist < min_dist):
                min_dist = dist
                closest_wp_idx = idx

        return closest_wp_idx       
        
        
        

    def get_light_state_ground_truth(self):
        """Determines the current color of the traffic light

        Args:
            self

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            int: Waypoint idex of closest traffic light 
 
        """
        state = TrafficLight.UNKNOWN
        light_wp = -1
           
        # Get Car position
        car_position_id = self.get_closest_waypoint(self.pose.pose, self.waypoints)

        # Get the closest Traffic Light Signal from the Car Laocation
        closest_tl_id = self.get_closest_waypoint(self.pose.pose, self.lights)

        # Check if Traffic light is ahead of the car
        if(car_position_id<closest_tl_id):

            # Get corresponding Traffic light
            closest_tl = self.lights[closest_tl_id]

            # Get the corresponding Stop line position from stop_line_positions
            closest_stop_line = self.config['stop_line_positions'][closest_tl_id]

            # Convert stop_line to pose object
            closest_stop_line_pose = Pose()
            closest_stop_line_pose.position.x = closest_stop_line[0]
            closest_stop_line_pose.position.y = closest_stop_line[1]

            # Get the waypoint id corresponding to the stop line
            light_wp = self.get_closest_waypoint(closest_stop_line_pose, self.waypoints)

            # get the state of the Closest Traffic light
            state = closest_tl.state


        return light_wp, state







    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            self

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            int: Waypoint idex of closest traffic light 

        """
        state = TrafficLight.UNKNOWN
        light_wp = -1
        
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification #TODO
        state = self.light_classifier.get_classification(cv_image)
        
        # Get Car position
        car_position_id = self.get_closest_waypoint(self.pose.pose, self.waypoints)

        # Get the closest Traffic Light Signal from the Car Laocation
        closest_tl_id = self.get_closest_waypoint(self.pose.pose, self.lights)

        # Check if Traffic light is ahead of the car
        if(car_position_id<closest_tl_id):

            # Get corresponding Traffic light
            closest_tl = self.lights[closest_tl_id]

            # Get the corresponding Stop line position from stop_line_positions
            closest_stop_line = self.config['stop_line_positions'][closest_tl_id]

            # Convert stop_line to pose object
            closest_stop_line_pose = Pose()
            closest_stop_line_pose.position.x = closest_stop_line[0]
            closest_stop_line_pose.position.y = closest_stop_line[1]

            # Get the waypoint id corresponding to the stop line
            light_wp = self.get_closest_waypoint(closest_stop_line_pose, self.waypoints)

        
        return light_wp, state
                                                 

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        state = TrafficLight.UNKNOWN
        light_wp = -1
                                         
        if(self.pose):
                                                 
            if self.use_tl_ground_truth:
                light_wp, state = self.get_light_state_ground_truth()
            else:
                light_wp, state = self.get_light_state()

        return light_wp, state

        

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

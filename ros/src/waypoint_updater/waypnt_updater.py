#!/usr/bin/env python
# File renamed to waypnt_updater.py because original name
# waypoint_updater.py which matches directory and package name
# prevented loading from waypoint_updater.cfg

import rospy
import math
import numpy as np
import itertools

import copy
from collections import deque

from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
from std_msgs.msg import String, Int32

from dynamic_reconfigure.server import Server
from waypoint_updater.cfg import DynReconfConfig


'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.  As mentioned in the doc, you should ideally first implement
a version which does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of
traffic lights too.

Please note that our simulator also provides the exact location of traffic
lights and their current status in `/vehicle/traffic_lights` message. You
can use this message to build this node as well as to verify your TL
classifier.

'''


# object to store details of motion at waypoint
class deq_waypoint(object):
    def __init__(self, ptr_id, waypoint):
        self.max_V = waypoint.twist.twist.linear.x
        self.V = 0.0
        self.A = 0.0
        self.J = 0.0
        self.waypoint = waypoint
        self.waypoint.twist.twist.linear.x = 0.0
        self.ptr_id = ptr_id
        self.state = None
        self.phase_start_ptr = 0  # initiate at ptr_id ?


class WaypointUpdater(object):
    def __init__(self):
        # set up variables

        # check on whether dynamic parameters have been received yet
        self.dyn_vals_received = False
        # list to hold waypoints from waypoint_loader
        self.waypoints = []
        # current reported pose
        self.pose = None
        # current reported velocity
        self.velocity = None
        # to store traffic light data in if required
        self.lights = None
        # number of waypoints to send out - not sure how many are useful
        self.lookahead_wps = 50  # 200
        # set up deque
        # have a backwards buffer
        self.rear_buffer = 20
        # set up max dequelen which will trigger right pops
        self.max_dequelen = self.lookahead_wps + self.rear_buffer
        # deque to hold final waypoints
        self.f_wpdeq = deque(maxlen=self.max_dequelen)
        # pointer to location in waypoints that final waypoints begins at
        self.final_waypoints_start_ptr = -1
        # closest waypoint used on last iteration when sending final waypoints
        self.last_waypoint = None
        # check to see if we found closest waypoint by looking backwards
        self.back_search = False
        # measurement of distance to closest waypoint on last iteration
        self.last_search_distance = None
        # time at which last closest_waypoint iteration occurred
        self.last_search_time = None
        # waypoint closest to next traffic light stop line when it is red
        self.next_traffic_light_wp = -1  # was None
        # speed to iterate next waypoint list
        self.update_rate = 10
        # global max velocity - this will be found in waypoints
        self.max_velocity = 0.0
        # default velocity is a dynamically adjustable target velocity
        self.default_velocity = 10.0
        # target max acceleration/braking force - dynamically adjustable
        self.default_accel = 1.5
        # are we in a deceleration phase
        self.slow_down = False
        # store jerk minimizing Trajectory coefficients here
        self.JMTcoeffs = None
        # where did velocity change start
        self.JMT_start_ptr = -1
        self.JMT_time = 0.0
        self.JMT_max_time = 0.0
        # dictionary of subscribers for this node
        self.subs = {}
        # dictionary of publishers for this node
        self.pubs = {}
        # dynamic reconfiguration server for this node
        self.dyn_reconf_srv = None
        # modes of operation
        self.states = ['stopped', 'startspeedup', 'JMTspeedup',
                       'maintainspeed', 'JMTslowdown']
        self.state = self.states[0]
        # variables defined

        rospy.init_node('waypoint_updater')

        # subscribe to topics
        # waypoints for trajectory
        self.subs['/base_waypoints'] = \
            rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # current pose of car
        self.subs['/current_pose'] = \
            rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # current speed of car
        self.subs['/current_velocity'] = \
            rospy.Subscriber('/current_velocity', TwistStamped,
                             self.velocity_cb)

        # If a red traffic light has been detected ahead
        # the waypoint closest to the stopline will be sent out
        # if no red light -1 will be sent
        self.subs['/traffic_waypoint'] = \
            rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint

        self.pubs['/final_waypoints'] = rospy.Publisher('/final_waypoints',
                                                        Lane, queue_size=1)

        # self.dyn_reconf_srv = Server(DynReconfConfig, self.dyn_vars_cb)
        # move to waypoints_cb so no collision - seems like a bad idea but
        # convenient for now

        self.loop()

    def loop(self):
        self.rate = rospy.Rate(self.update_rate)
        # wait for waypoints and pose to be loaded before trying to update
        # waypoints
        while not self.waypoints:
            self.rate.sleep()
        while not self.pose:
            self.rate.sleep()
        while not rospy.is_shutdown():

            if self.waypoints:
                self.send_waypoints()
            self.rate.sleep()

    def velocity_cb(self, twist_msg):
        self.velocity = twist_msg

    def pose_cb(self, pose_msg):
        self.pose = pose_msg.pose
        rospy.logdebug("waypoint_updater:pose_cb pose set to  %s", self.pose)

    # Receive a msg from /traffic_waypoint about the next stop line
    def traffic_cb(self, traffic_msg):
        if traffic_msg.data != self.next_traffic_light_wp:
            self.next_traffic_light_wp = traffic_msg.data
            rospy.loginfo("new /traffic_waypoint message received at wp: %d."
                          "while car is at wp %d", self.next_traffic_light_wp,
                          self.final_waypoints_start_ptr)
        else:
            # just for debug to see what we're getting
            rospy.logdebug("same /traffic_waypoint message received.")
        # end if else

    def send_waypoints(self):
        # generates the list of LOOKAHEAD_WPS waypoints based on car location
        # for now assume waypoints form a loop
        # waypoints loaded into a deque with deepcopy
        # buffer of rear_buffer points to use for location search
        # final_waypoints_start_ptr refers to waypoint_ptr which will be start
        # of published final_waypoints list
        # last_waypoint was the initial waypoint in previous iteration

        start_wps_ptr = self.closest_waypoint()
        dequelen = len(self.f_wpdeq)
        if self.final_waypoints_start_ptr < 0:
            # condition on startup
            self.final_waypoints_start_ptr = start_wps_ptr
            self.last_waypoint = self.final_waypoints_start_ptr
            end_waypoint_in_deque = self.last_waypoint - 1
            # load buffer of points behind into deque
            for w_ptr in range(end_waypoint_in_deque - self.rear_buffer,
                               end_waypoint_in_deque+1):
                wpd = deq_waypoint(w_ptr, copy.deepcopy(self.waypoints[w_ptr]))
                self.f_wpdeq.append(wpd)
        else:
            self.last_waypoint = self.final_waypoints_start_ptr
            end_waypoint_in_deque = self.f_wpdeq[dequelen - 1].\
                ptr_id
            self.final_waypoints_start_ptr = start_wps_ptr
        # end if else

        points_added = 0
        last_waypoint_needed = (self.final_waypoints_start_ptr +
                                self.lookahead_wps - 1) % len(self.waypoints)
        if last_waypoint_needed == end_waypoint_in_deque:
            rospy.loginfo("no movement, not adding to deque")
        elif last_waypoint_needed > end_waypoint_in_deque:
            for w_ptr in range(end_waypoint_in_deque+1,
                               last_waypoint_needed+1):
                wpd = deq_waypoint(w_ptr, copy.deepcopy(self.waypoints[w_ptr]))
                self.f_wpdeq.append(wpd)
                points_added += 1
            # end of for
        else:
            for w_ptr in range(end_waypoint_in_deque+1, len(self.waypoints)):
                # w_p.twist.twist.linear.x = self.default_velocity
                wpd = deq_waypoint(w_ptr, copy.deepcopy(self.waypoints[w_ptr]))
                self.f_wpdeq.append(wpd)
                points_added += 1
            # end of for
            for w_ptr in range(last_waypoint_needed+1):
                # w_p.twist.twist.linear.x = self.default_velocity
                wpd = deq_waypoint(w_ptr, copy.deepcopy(self.waypoints[w_ptr]))
                self.f_wpdeq.append(wpd)
                points_added += 1
            # end of for
        # end of if
        # #for ptr in range(end_waypoint_in_deque+1, last_waypoint_needed):
        # self.f_wpdeq.append(copy.deepcopy(self.waypoints
        # [end_waypoint_in_deque+1:last_waypoint_needed]))
        rospy.loginfo("waypoint_updater:send_waypoints start_wps_ptr = %d,"
                      " end_wps_ptr = %d, added %d points last item ptr = %d", start_wps_ptr,
                      last_waypoint_needed, points_added, self.f_wpdeq[len(self.f_wpdeq)-1].ptr_id)

        self.set_waypoints_velocity()
        # rospy.loginfo("waypoint_updater:send_waypoints final_waypoints list"
        #              " length is %d", len(new_wps_list))
        lane = Lane()
        
        new_wps_list = []
        for wpt in list(itertools.islice(self.f_wpdeq,
                        self.rear_buffer, self.max_dequelen)):
            new_wps_list.append(wpt.waypoint)
            # rospy.loginfo("new_wps_list[{}].twist.twist.linear.x = {}".\
            #               format(wpt.ptr_id, wpt.waypoint.twist.twist.linear.x))
        # end for
        # rospy.loginfo("new_wps_list len = {}".format(len(new_wps_list)))
        lane.waypoints = list(new_wps_list)
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        self.pubs['/final_waypoints'].publish(lane)

    def set_waypoints_velocity(self):
        # adjust the velocities in the /final_waypoint queue
        # TODO: move the waypoints to a structure there first so don't have to
        # worry about looping
        disp = 0  # set for logging

        start_ptr = self.rear_buffer
        last_wpd = self.f_wpdeq[start_ptr-1]
        # for wpd in list(itertools.islice(self.f_wpdeq,
        #  start_ptr,
        #                                 self.max_dequelen)):

        for l_ptr in range(start_ptr, self.max_dequelen):
            if self.f_wpdeq[l_ptr].state in self.states:
                self.state = self.f_wpdeq[l_ptr].state
            # wpd = self.f_wpdeq[l_ptr]
            calc = self.state_spinner(l_ptr-1, l_ptr)

            if calc is True:

                disp = self.distance(self.JMT_start_ptr, self.f_wpdeq[l_ptr].ptr_id)

                if self.state == 'stopped':
                    self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = 0.0
                    self.f_wpdeq[l_ptr].V = 0.0
                    self.f_wpdeq[l_ptr].A = 0.0
                    self.f_wpdeq[l_ptr].J = 0.0
                # end if

                if self.state == 'maintainspeed':
                    v = min(self.f_wpdeq[l_ptr].max_V, last_wpd.V)
                    self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = v
                    self.f_wpdeq[l_ptr].V = v
                    self.f_wpdeq[l_ptr].A = 0.0
                    self.f_wpdeq[l_ptr].J = 0.0
                # end if

                if self.state == 'JMTslowdown':              
                    wp_check = self.JMTD_at(disp, self.JMTcoeffs, self.JMT_time,
                                            self.JMT_max_time, l_ptr)
                    if wp_check == -1:
                        rospy.logwarn("JMTslowdown error - switch to maintainspeed")
                        self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = last_wpd.V
                        self.f_wpdeq[l_ptr].V = last_wpd.V
                        self.state = 'maintainspeed'
                    else:
                        if self.f_wpdeq[l_ptr].V < 0.25:
                            self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = 0.0
                        else:
                            self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = \
                                min(self.f_wpdeq[l_ptr].V, self.f_wpdeq[l_ptr].max_V)
                        # end for
                    # end if else
                # end if

                if self.state == 'JMTspeedup':
                    wp_check = self.JMTD_at(disp, self.JMTcoeffs, self.JMT_time,
                                            self.JMT_max_time, l_ptr)
                    if wp_check == -1:
                        self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = \
                            min(last_wpd.V, self.f_wpdeq[l_ptr].max_V)
                        self.state = 'maintainspeed'  # Assume gone past end
                        self.f_wpdeq[l_ptr].state = self.state
                        self.JMTcoeffs = None
                        self.JMT_start_ptr = self.f_wpdeq[l_ptr].ptr_id
                        self.JMT_time = 0.0
                        self.JMT_max_time = 0.0

                    else:
                        self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = \
                            min(self.f_wpdeq[l_ptr].V, self.f_wpdeq[l_ptr].max_V)

                if self.state == 'startspeedup':
                    # dist = self.distance(last_wpd.ptr_id, self.f_wpdeq[l_ptr].ptr_id)
                    self.f_wpdeq[l_ptr].A = self.default_accel
                    v = max(0.5 , math.sqrt(self.f_wpdeq[l_ptr].A * disp * 2.0))
                    v = min(v, self.f_wpdeq[l_ptr].max_V)
                    self.f_wpdeq[l_ptr].V = v
                    self.f_wpdeq[l_ptr].waypoint.twist.twist.linear.x = v

                last_wpd = self.f_wpdeq[l_ptr]
                rospy.loginfo("state = {}, ptr= {}, v= {}, a= {}, dist= {}".
                            format(self.state, last_wpd.ptr_id, last_wpd.V,
                                    last_wpd.A, disp))
            else:
                disp = self.distance(self.f_wpdeq[l_ptr].phase_start_ptr, self.f_wpdeq[l_ptr].ptr_id)
                last_wpd = self.f_wpdeq[l_ptr]
                rospy.loginfo("skipping recalculation "\
                              "state = {}, ptr= {}, v= {}, a= {}, dist= {}".
                              format(self.state, last_wpd.ptr_id, last_wpd.V,
                                     last_wpd.A, disp))
                

    # check for state transitions
    def state_spinner(self, last_dwp_ptr, dwp_ptr):

        calc = True  # set to know if v has to be recalculated

        last_deq_waypoint = self.f_wpdeq[last_dwp_ptr]
        deq_waypoint = self.f_wpdeq[dwp_ptr]
        if deq_waypoint.state is None:
            if self.state not in ['JMTslowdown', 'stopped']:

                if self.next_traffic_light_wp == -1:
                    dist = 1000  # far away
                else:
                    dist = self.distance(deq_waypoint.ptr_id,
                                         self.next_traffic_light_wp)

                if dist < self.get_stopping_distance(last_deq_waypoint.V):
                    self.state = 'JMTslowdown'
                    # setup JMT for calcs

                    end_wp = self.next_traffic_light_wp - 1
                    dist = self.distance(deq_waypoint.ptr_id, end_wp)
                    # try to bias to slowing down from start by setting A slightly negative
                    start = list([0, last_deq_waypoint.V, last_deq_waypoint.A-0.1])
                    end = list([dist, 0.0, 0.0])
                    time = 2.0 * 2 * dist / last_deq_waypoint.V
                    self.JMTcoeffs = self.JMT(start, end, time)
                    self.JMT_start_ptr = deq_waypoint.ptr_id
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id
                    self.JMT_time = 0.0
                    self.JMT_max_time = time * 1.1
                    rospy.logwarn("Start slowing down! Wpt = {},"
                                  " start = {}, {}, {} "
                                  "end = {}, {}, {} at T= {}".format(
                                  deq_waypoint.ptr_id,
                                  start[0], start[1], start[2],
                                  end[0], end[1], end[2], time))

                # end if
            # end if

            if self.state == 'startspeedup':
                if last_deq_waypoint.V >= 1.5:
                    self.state = 'JMTspeedup'
                    d1 = self.distance(last_deq_waypoint.ptr_id,
                                    deq_waypoint.ptr_id)
                    self.f_wpdeq[dwp_ptr].V = math.sqrt(last_deq_waypoint.V**2 + 2 *
                                            self.default_accel * d1)
                    self.f_wpdeq[dwp_ptr].A = self.default_accel
                    time = 2.0 * (self.default_velocity - self.f_wpdeq[dwp_ptr].V) / self.default_accel
                    dist = (self.default_velocity - self.f_wpdeq[dwp_ptr].V) * time
                    start = list([0, self.f_wpdeq[dwp_ptr].V, self.f_wpdeq[dwp_ptr].A])
                    end = list([dist, self.default_velocity * 0.975, 0.0])
                    self.JMTcoeffs = self.JMT(start, end, time)
                    self.JMT_start_ptr = deq_waypoint.ptr_id
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id
                    self.JMT_time = 0.0
                    self.JMT_max_time = time * 1.1
                    rospy.logwarn("Start JMT phase of accel! Wpt = {},"
                                  " start = {}, {}, {} "
                                  "end = {}, {}, {} at T= {}".format(
                                  deq_waypoint.ptr_id,
                                  start[0], start[1], start[2],
                                  end[0], end[1], end[2], time))

                # end if
            # end if

            elif self.state == 'JMTspeedup':
                if last_deq_waypoint.V >= self.default_velocity * 0.975:
                    # and (last_deq_waypoint.A < 0.3)):
                    self.state = 'maintainspeed'
                    rospy.logwarn("Switch to maintainspeed. Wpt = {}".format(
                                deq_waypoint.ptr_id))
                    self.JMTcoeffs = None
                    self.JMT_start_ptr = self.f_wpdeq[dwp_ptr].ptr_id - 1
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id - 1
                    self.JMT_time = 0.0
                    self.JMT_max_time = 0.0
                # end if
            # end if

            elif self.state == 'JMTslowdown':
                if last_deq_waypoint.V <= 0.2:
                    self.f_wpdeq[dwp_ptr].V = 0.0
                    self.f_wpdeq[dwp_ptr].A = 0.0
                    self.state = 'stopped'
                    rospy.logwarn("Switch to stopped! Wpt = {}".format(
                                deq_waypoint.ptr_id))
                    self.JMTcoeffs = None
                    self.JMT_start_ptr = self.f_wpdeq[dwp_ptr].ptr_id
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id
                    self.JMT_time = 0.0
                    self.JMT_max_time = 0.0
                # end if
            # end if

            elif self.state == 'stopped':
                if (self.distance(deq_waypoint.ptr_id,
                                self.next_traffic_light_wp) > 10.0):
                    self.state = 'startspeedup'
                    self.JMT_start_ptr = self.f_wpdeq[dwp_ptr].ptr_id
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id
                    rospy.logwarn("Time to start moving! Wpt = {}".format(
                                deq_waypoint.ptr_id))
                # end if
            # end if
            self.f_wpdeq[dwp_ptr].phase_start_ptr = self.JMT_start_ptr

        else:  # state set on waypoint - we've already calculated it

            self.state = self.f_wpdeq[dwp_ptr].state
            if self.next_traffic_light_wp == -1:
                dist = 1000
            else:
                dist = self.distance(deq_waypoint.ptr_id,
                                     self.next_traffic_light_wp)

            if self.state not in ['stopped', 'JMTslowdown']:

                if dist < self.get_stopping_distance(last_deq_waypoint.V):
                    self.state = 'JMTslowdown'
                    # setup JMT for calcs

                    end_wp = self.next_traffic_light_wp - 1
                    dist2 = self.distance(deq_waypoint.ptr_id, end_wp)
                    start = list([0, last_deq_waypoint.V, last_deq_waypoint.A - 0.1])
                    end = list([dist2, 0.0, 0.0])
                    time = 2.0 * 2.0 * dist / last_deq_waypoint.V
                    # time = 1.5 * (last_deq_waypoint.V / self.default_accel)
                    self.JMTcoeffs = self.JMT(start, end, time)
                    self.JMT_start_ptr = deq_waypoint.ptr_id - 1
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id - 1
                    self.JMT_time = 0.0
                    self.JMT_max_time = time * 1.1
                    rospy.logwarn("Reset to slowing down! Wpt = {},"
                                  " start = {}, {}, {} "
                                  "end = {}, {}, {} at T= {}".format(
                                  deq_waypoint.ptr_id,
                                  start[0], start[1], start[2],
                                  end[0], end[1], end[2], time))
                # end if
            # end if

            elif self.state == 'stopped':
                if dist > 10.0:
                    self.state = 'startspeedup'
                    self.JMT_start_ptr = self.f_wpdeq[dwp_ptr].ptr_id
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id - 1
                    rospy.logwarn("Time to start moving! Wpt = {}".format(
                                deq_waypoint.ptr_id))
                # end if
            # end if

            elif self.state == 'JMTslowdown':
                if dist > 130.0:
                    self.state = 'JMTspeedup'
                    rospy.logwarn("Switch to JMTspeedup! Wpt = {}".format(
                                deq_waypoint.ptr_id))
                    d1 = self.distance(last_deq_waypoint.ptr_id,
                                    deq_waypoint.ptr_id)
                    self.f_wpdeq[dwp_ptr].V = math.sqrt(last_deq_waypoint.V**2 + 2 *
                                            last_deq_waypoint.A * d1)
                    self.f_wpdeq[dwp_ptr].A = last_deq_waypoint.A
                    time = 2.0 * (self.default_velocity - self.f_wpdeq[dwp_ptr].V) / self.default_accel
                    dist = (self.default_velocity - self.f_wpdeq[dwp_ptr].V) * time
                    start = list([0, self.f_wpdeq[dwp_ptr].V, self.f_wpdeq[dwp_ptr].A])
                    end = list([dist, self.default_velocity, 0.0])
                    self.JMTcoeffs = self.JMT(start, end, time)
                    self.f_wpdeq[dwp_ptr].phase_start_ptr = deq_waypoint.ptr_id
                    self.JMT_start_ptr = deq_waypoint.ptr_id
                    self.JMT_time = 0.0
                    self.JMT_max_time = time * 1.1
                    rospy.logwarn("Switch to start speeding up! Wpt = {},"
                                  " start = {}, {}, {} "
                                  "end = {}, {}, {} at T= {}".format\
                                  (deq_waypoint.ptr_id,
                                   start[0], start[1], start[2],
                                   end[0], end[1], end[2], time))

                # end if
            # end if

            # reset state on all remaining items in deque if we need to recalc traj
            if self.f_wpdeq[dwp_ptr].state != self.state:
                self.f_wpdeq[dwp_ptr].state = self.state
                for l_ptr2 in range(dwp_ptr+1, self.max_dequelen-1):
                        self.f_wpdeq[l_ptr2].state = None
                        self.f_wpdeq[l_ptr2].phase_start_ptr = 0
            else:
                calc = False

        self.f_wpdeq[dwp_ptr].state = self.state
        return calc

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
        # We will implement it later
        pass

    # Load set of waypoints from /basewaypoints into self.waypoints
    # this should only happen once, so we unsubscribe at end
    def waypoints_cb(self, lane_msg):
        rospy.loginfo("waypoint_updater:waypoints_cb loading waypoints")
        if not self.waypoints:
            max_velocity = 0.0
            for waypoint in lane_msg.waypoints:
                self.waypoints.append(waypoint)
                if max_velocity < self.get_waypoint_velocity(waypoint):
                    max_velocity = self.get_waypoint_velocity(waypoint)
                # end if
            rospy.loginfo("waypoint_updater:waypoints_cb %d waypoints loaded",
                          len(self.waypoints))
            # setting max velocity based on project requirements in
            # Waypoint Updater Node Revisited
            self.max_velocity = max_velocity
            rospy.loginfo("waypoint_updater:waypoints_cb max_velocity set to "
                          " {} based on max value in waypoints."
                          .format(self.max_velocity))
            # now max_velocity is known, set up dynamic reconfig
            if not self.dyn_reconf_srv:
                self.dyn_reconf_srv = Server(DynReconfConfig, self.dyn_vars_cb)
                rospy.loginfo("dynamic_parm server started")
            # end if
        else:
            rospy.logerr("waypoint_updater:waypoints_cb attempt to load "
                         "waypoints when we have already loaded %d waypoints",
                         len(self.waypoints))
        # end if else
        self.subs['/base_waypoints'].unregister()
        rospy.loginfo("Unregistered from /base_waypoints topic")

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypointlist, waypoint, velocity):
        waypointlist[waypoint].twist.twist.linear.x = velocity

    def closest_waypoint(self):
        # TODO:
        # Require 2+ consecutive increases to end search in case of pose noise
        # Make search more generic and switch order of search direction based
        #  on last search direction
        def distance_lambda(a, b): return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2)
        if self.f_wpdeq:
            dist = distance_lambda(self.f_wpdeq[self.
                                   rear_buffer-1].waypoint.pose.pose.position,
                                   self.pose.position)
            # if self.back_search is True:
            for i in range(self.rear_buffer, self.max_dequelen):
                tmpdist = distance_lambda(self.f_wpdeq[i].
                                          waypoint.pose.pose.position,
                                          self.pose.position)

                if tmpdist < dist:
                    dist = tmpdist
                else:
                    # distance is starting to get larger so look at
                    # last position
                    if (i == self.rear_buffer+1):
                        # we're closest to original waypoint, but what if
                        # we're going backwards - loop backwards to make sure
                        # a point further back  isn't closest
                        for j in range(self.rear_buffer - 1, 0, -1):
                            tmpdist = \
                                distance_lambda(self.f_wpdeq[j].
                                                waypoint.pose.pose.position,
                                                self.pose.position)

                            if tmpdist < dist:
                                dist = tmpdist
                                self.back_search = True
                            else:
                                if abs(dist-self.last_search_distance) < 5.0:
                                    self.last_search_distance = dist
                                    return self.f_wpdeq[j+1].\
                                        ptr_id
                                else:
                                    break
                            # end if else
                        # end for - reverse loop
                    # end if

                    if abs(dist-self.last_search_distance) < 5.0:
                        self.last_search_distance = dist
                        return self.f_wpdeq[i-1].ptr_id
                    # end if
                # end if else
            # end for - fall out no closest match that looks acceptable
            rospy.logwarn("waypoint_updater:closest_waypoint local search not"
                          "satisfied - run full search")
        # end if

        dist = 1000000  # maybe should use max
        closest = 0
        for i in range(len(self.waypoints)):
            tmpdist = distance_lambda(self.waypoints[i].pose.pose.position,
                                      self.pose.position)
            if tmpdist < dist:
                closest = i
                dist = tmpdist
            # end of if
        # end of for
        self.last_search_distance = dist
        return closest
        # Note: the first waypoint is closest to the car, not necessarily in
        # front of it.  Waypoint follower is responsible for finding this

    def dyn_vars_cb(self, config, level):
        # adjust dynamic variables

        self.dyn_vals_received = True
        if self.update_rate:
            old_update_rate = self.update_rate
            old_default_velocity = self.default_velocity
            old_default_accel = self.default_accel
            old_lookahead_wps = self.lookahead_wps
            old_test_stoplight_wp = self.next_traffic_light_wp
        # end if

        rospy.logdebug("Received dynamic parameters {} with level: {}"
                       .format(config, level))

        # TODO:could probably replace a lot of this code by setting up a list
        # of dynamic params, iterating through it, and then running any actions
        if old_update_rate != config['dyn_update_rate']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting update Rate "
                          "from {} to {}".format(old_update_rate,
                                                 config['dyn_update_rate']))
            self.update_rate = config['dyn_update_rate']
            # need to switch the delay
            self.rate = rospy.Rate(self.update_rate)
        # end if

        if old_default_velocity != config['dyn_default_velocity']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting default_"
                          "velocity from {} to {}"
                          .format(old_default_velocity,
                                  config['dyn_default_velocity']))

            if config['dyn_default_velocity'] > self.max_velocity:
                rospy.logwarn("waypoint_updater:dyn_vars_cb default_velocity "
                              "limited to max_velocity {}"
                              .format(self.max_velocity))
                self.default_velocity = self.max_velocity
            else:
                self.default_velocity = config['dyn_default_velocity']
            # end if
        # end if

        if old_lookahead_wps != config['dyn_lookahead_wps']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting lookahead_"
                          "wps from {} to {}"
                          .format(old_lookahead_wps,
                                  config['dyn_lookahead_wps']))
            self.lookahead_wps = config['dyn_lookahead_wps']
        # end if

        if old_test_stoplight_wp != config['dyn_test_stoplight_wp']:
            rospy.logwarn("waypoint_updater:dyn_vars_cb Adjusting next "
                          "stoplight from {} to {}"
                          .format(old_test_stoplight_wp,
                                  config['dyn_test_stoplight_wp']))
            self.next_traffic_light_wp = config['dyn_test_stoplight_wp']
            self.slow_down = False  # reset
        # end if

        if old_default_accel != config['dyn_default_accel']:
            rospy.logwarn("waypoint_updater:dyn_vars_cb Adjusting default_"
                          "accel from {} to {}"
                          .format(old_default_accel,
                                  config['dyn_default_accel']))
            self.default_accel = config['dyn_default_accel']

        # we can also send adjusted values back
        return config

    def distance(self, wp1, wp2):
        # TODO need to adjust for looping or reverse

        dist = 0
        if wp1 == wp2:
            return 0.0

        def distance_lambda(a, b): return math.sqrt((a.x-b.x)**2 +
                                                    (a.y-b.y)**2 +
                                                    (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += distance_lambda(self.waypoints[wp1].pose.pose.position,
                                    self.waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_stopping_distance(self, velocity):
        # reasonable guess of distance needed to stop at speed and
        # max decceleration rate

        safety_factor = 1.2  # this should be much smaller with braking
        distance_needed = velocity * velocity / (2.0 * self.default_accel)
        distance_needed *= safety_factor
        # if distance_needed > 0.0 and distance_needed < 75.0:
        #    rospy.loginfo("Stopping Distance Needed is {}".
        #                  format(distance_needed))
        return distance_needed

    def JMT(self, start, end, T):
        """
        Calculates Jerk Minimizing Trajectory for start, end and T.
        """
        a_0, a_1, a_2 = start[0], start[1], start[2] / 2.0
        c_0 = a_0 + a_1 * T + a_2 * T**2
        c_1 = a_1 + 2 * a_2 * T
        c_2 = 2 * a_2

        A = np.array([
                     [T**3,   T**4,    T**5],
                     [3*T**2, 4*T**3,  5*T**4],
                     [6*T,   12*T**2, 20*T**3],
                     ])

        B = np.array([
                     end[0] - c_0,
                     end[1] - c_1,
                     end[2] - c_2
                     ])
        a_3_4_5 = np.linalg.solve(A, B)
        alphas = np.concatenate([np.array([a_0, a_1, a_2]), a_3_4_5])
        return alphas

    def JMTD_at(self, displacement, coeffs, t0, tmax, deq_wpt_ptr):
        # find JMT descriptors at displacement
        s_last = 0.0
        t_found = False
        t_inc = 0.005

        for t_cnt in range(int((tmax-t0)/t_inc) + 1):

            t = t0 + t_cnt * t_inc
            s = coeffs[0] + coeffs[1]*t + coeffs[2]*t*t + coeffs[3]*t*t*t +\
                coeffs[4]*t*t*t*t + coeffs[5]*t*t*t*t*t
            if s > displacement:
                t = t - (1 - (s - displacement) / (s - s_last)) * t_inc
                t_found = True
                break
            # end if
            s_last = s
        # end for
        if t_found is False:
            rospy.logerr("waypoint_updater:JMTDat Ran out of bounds without "
                         "finding target displacement")
            return(-1)
        self.JMT_time = t
        t2 = t*t
        t3 = t2*t
        t4 = t3*t
        t5 = t4*t

        s = coeffs[0] + coeffs[1]*t + coeffs[2]*t2 + coeffs[3]*t3 +\
            coeffs[4]*t4 + coeffs[5]*t5
        v = coeffs[1] + 2*coeffs[2]*t +\
            3*coeffs[3]*t2 + 4*coeffs[4]*t3 + 5*coeffs[5]*t4
        a = 2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t2 +\
            20*coeffs[5]*t3
        j = 6*coeffs[3] + 24*coeffs[4]*t + 60*coeffs[5]*t2

        delta_s = (displacement - s)
        if delta_s > 0.1:
            rospy.logwarn("waypoint_updater:JMTD_at need to refine algo,"
                          " delta_s is {}".format(delta_s))

        self.f_wpdeq[deq_wpt_ptr].waypoint.twist.twist.linear.x = v
        self.f_wpdeq[deq_wpt_ptr].V = v
        self.f_wpdeq[deq_wpt_ptr].A = a
        self.f_wpdeq[deq_wpt_ptr].J = j 

        rospy.loginfo("waypoint_updater:JMTD_at for displacement {} found s= {}, at t= {}, v={}, a={}"
                      .format(displacement, s, t, v, a))

        return v


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

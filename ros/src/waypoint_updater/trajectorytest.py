#!/usr/bin/env python
import math
import numpy as np


def main():
    test = traj()
    test.load_waypoints()
    test.set_location(20)
    test.generate_trajectory()


def get_accel_distance(Vi, Vf, A):
    return (Vf**2 - Vi**2)/(2.0 * A)


def get_accel_time(S, Vi, Vf):
    return(2.0 * S / (Vi + Vf))


class JMT(object):
    def __init__(self, start, end, T):
        """
        Calculates Jerk Minimizing Trajectory for start, end and T.
        start and end include
        [displacement, velocity, acceleration]
        """
        self.start = start
        self.end = end
        self.final_displacement = end[0]
        self.T = T

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
        self.coeffs = np.concatenate([np.array([a_0, a_1, a_2]), a_3_4_5])

    # def JMTD_at(self, displacement, coeffs, t0, tmax, deq_wpt_ptr):
    def JMTD_at(self, displacement, t0, tmax):
        # find JMT descriptors at displacement
        s_last = 0.0
        t_found = False
        t_inc = 0.005

        for t_cnt in range(int((tmax-t0)/t_inc) + 1000):

            t = t0 + t_cnt * t_inc
            s = self.coeffs[0] + self.coeffs[1]*t + self.coeffs[2]*t*t + self.coeffs[3]*t*t*t +\
                self.coeffs[4]*t*t*t*t + self.coeffs[5]*t*t*t*t*t
            if s > displacement:
                t = t - (1 - (s - displacement) / (s - s_last)) * t_inc
                t_found = True
                break
            # end if
            s_last = s
        # end for
        if t_found is False:
            print("waypoint_updater:JMTDat Ran out of bounds without "
                         "finding target displacement")
            return None

        t2 = t*t
        t3 = t2*t
        t4 = t3*t
        t5 = t4*t

        s = self.coeffs[0] + self.coeffs[1]*t + self.coeffs[2]*t2 + self.coeffs[3]*t3 +\
            self.coeffs[4]*t4 + self.coeffs[5]*t5
        v = self.coeffs[1] + 2*self.coeffs[2]*t +\
            3*self.coeffs[3]*t2 + 4*self.coeffs[4]*t3 + 5*self.coeffs[5]*t4
        a = 2*self.coeffs[2] + 6*self.coeffs[3]*t + 12*self.coeffs[4]*t2 +\
            20*self.coeffs[5]*t3
        j = 6*self.coeffs[3] + 24*self.coeffs[4]*t + 60*self.coeffs[5]*t2

        delta_s = (displacement - s)
        if delta_s > 0.1:
            print("waypoint_updater:JMTD_at need to refine algo,"
                          " delta_s is {}".format(delta_s))

        
        details = JMTDetails(s,v,a,j,t)

        print ("waypoint_updater:JMTD_at displacement {} found s,v,a,j,t = {}".format(displacement, details))

        return details


class pose(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class JMTDetails(object):
    def __init__(self, S, V, A, J, t):
        self.S = S
        self.V = V
        self.A = A
        self.J = J
        self.time = t

    def set_VAJt(self, V, A, J, time):
        self.V = V
        self.A = A
        self.J = J
        self.time = time

    def __repr__(self):
        return "%5.3f, %2.4f, %2.4f, %2.4f, %2.3f" % (self.S, self.V, self.A, self.J, self.time)


class waypoint(object):
    def __init__(self, xval, yval, zval, max_v, ptr_id, s):
        self.position = pose(xval, yval, zval)
        self.v = max_v
        self.JMTD = JMTDetails(s,0.0,0.0,0.0,0.0)
        self.ptr_id = ptr_id
        self.state = None
        self.JMT_ptr = -1 # points to JMT object


class traj(object):
    def __init__(self):

        # set up variables
        self.waypoints = []
        # pointer to location in waypoints that final waypoints begins at
        self.final_waypoints_start_ptr = -1
        # closest waypoint used on last iteration when sending final waypoints
        self.last_waypoint = None
        #
        self.max_velocity = 11.1
        # default velocity is a dynamically adjustable target velocity
        self.default_velocity = 11.0
        self.max_A = 9.0
        self.max_V = 11.1
        self.max_J = 9.0
        # target max acceleration/braking force - dynamically adjustable
        self.default_accel = 1.5
        # are we in a deceleration phase
        self.slow_down = False
        # store details of JMT phases in a list
        self.JMT_List = []
        # store jerk minimizing Trajectory coefficients here
        self.JMTcoeffs = None
        # where did velocity change start
        # self.JMT_start_ptr = -1
        # self.JMT_time = 0.0
        # self.JMT_max_time = 0.0
        # modes of operation
        self.states = ['stopped', 'startspeedup', 'JMTspeedup',
                       'maintainspeed', 'JMTslowdown']
        self.state = self.states[0]

        self.initiate_waypoints()

    def load_waypoints(self):
        cntr = 0
        s = 0.0
        for pt in self.raw_waypoints:
            
            if cntr > 0:
                # won't come into here until after wpt loaded
                # in previous loop
                s += math.sqrt((wpt.position.x-pt[0])**2 +
                               (wpt.position.y-pt[1])**2)

            wpt = waypoint(pt[0],pt[1],pt[2],pt[3],cntr,s)
            self.waypoints.append(wpt)
            cntr += 1
            
        print "Loaded {} waypoints".format(len(self.waypoints))

    def set_location(self, ptr_id):
        self.final_waypoints_start_ptr = ptr_id

    def generate_trajectory(self):
        ptr = self.final_waypoints_start_ptr
        curpt = self.waypoints[ptr]
        curpt.JMTD.V = 1.5
        curpt.JMTD.A = self.default_accel

        target_velocity = self.max_velocity * 0.9

        jmt_ptr = self.setup_jmt(curpt, target_velocity)

        curpt.JMT_ptr = jmt_ptr
        JMT_instance = self.JMT_List[jmt_ptr]

        t = 0.0
        last_wpt = ptr+1
        recalc = False
        for wpt in self.waypoints[ptr+1:]:
            last_wpt = wpt.ptr_id
            if wpt.JMTD.S > JMT_instance.final_displacement:
                print "Passed beyond S = {} at ptr_id = {}".format(JMT_instance.final_displacement, last_wpt)
                break
            jmt_point = JMT_instance.JMTD_at(wpt.JMTD.S, t, JMT_instance.T*1.5)
            if jmt_point is None:
                print "JMT_at returned None at ptr_id = {}".format(last_wpt)
                break

            wpt.JMTD.set_VAJt(jmt_point.V, jmt_point.A, jmt_point.J, jmt_point.time)
            
            if jmt_point.A > self.max_A:
                print "A of {} exceeds max value of {} at ptr = {}".format(jmt_point.A, self.max_A, wpt.ptr_id)
                recalc = True
            if jmt_point.J > self.max_J:
                print "J of {} exceeds max value of {} at ptr = {}".format(jmt_point.J, self.max_J, wpt.ptr_id)
                recalc = True
            if jmt_point.V > self.max_V:
                print "V of {} exceeds max value of {} at ptr = {}".format(jmt_point.V, self.max_V, wpt.ptr_id)
                recalc = True
        print "ptr_id, S, V, A, J, time"
        for wpt in self.waypoints[ptr:last_wpt]:
            print "{}, {}".format(wpt.ptr_id, wpt.JMTD)

        print "recalc is set to {} ".format(recalc)


    def setup_jmt(self, curpt, target_velocity):

        a_dist = get_accel_distance(curpt.JMTD.V, target_velocity, self.default_accel)
        T = get_accel_time(a_dist, curpt.JMTD.V, target_velocity)

        print "Setup JMT to accelerate to {} by distance {} in time {}".format(target_velocity, a_dist, T)

        start = [curpt.JMTD.S, curpt.JMTD.V, curpt.JMTD.A]
        end = [curpt.JMTD.S + a_dist, target_velocity, 0.0]
        jmt = JMT(start, end, T)
        self.JMT_List.append(jmt)
        jmt_ptr = len(self.JMT_List)

        return jmt_ptr-1


    def distance(self, wp1, wp2):
        # TODO need to adjust for looping or reverse
        dist = 0
        if wp1 == wp2:
            return 0.0

        def distance_lambda(a, b): return math.sqrt((a.x-b.x)**2 +
                                                    (a.y-b.y)**2 +
                                                    (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += distance_lambda(self.waypoints[wp1].position,
                                    self.waypoints[i].position)
            # dist += distance_lambda(self.waypoints[wp1].pose.pose.position,
            #                         self.waypoints[i].pose.pose.position)
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



    def initiate_waypoints(self):
        self.raw_waypoints = [[1116.28,1181.84,0.0,11.1],
            [1117.15,1181.95,0.0,11.1],
            [1118.03,1182.07,0.0,11.1],
            [1118.9,1182.18,0.0,11.1],
            [1119.78,1182.29,0.0,11.1],
            [1120.22,1182.35,0.0,11.1],
            [1121.1,1182.45,0.0,11.1],
            [1121.97,1182.55,0.0,11.1],
            [1122.85,1182.65,0.0,11.1],
            [1123.73,1182.75,0.0,11.1],
            [1124.6,1182.84,0.0,11.1],
            [1125.04,1182.89,0.0,11.1],
            [1125.92,1182.98,0.0,11.1],
            [1126.8,1183.07,0.0,11.1],
            [1127.68,1183.16,0.0,11.1],
            [1128.56,1183.24,0.0,11.1],
            [1129.44,1183.33,0.0,11.1],
            [1129.88,1183.37,0.0,11.1],
            [1130.76,1183.45,0.0,11.1],
            [1131.64,1183.53,0.0,11.1],
            [1132.52,1183.6,0.0,11.1],
            [1133.4,1183.68,0.0,11.1],
            [1134.28,1183.75,0.0,11.1],
            [1135.16,1183.82,0.0,11.1],
            [1136.04,1183.89,0.0,11.1],
            [1136.48,1183.93,0.0,11.1],
            [1137.37,1184,0.0,11.1],
            [1138.25,1184.07,0.0,11.1],
            [1139.13,1184.13,0.0,11.1],
            [1140.02,1184.19,0.0,11.1],
            [1140.9,1184.26,0.0,11.1],
            [1141.78,1184.32,0.0,11.1],
            [1142.66,1184.38,0.0,11.1],
            [1143.1,1184.41,0.0,11.1],
            [1143.98,1184.47,0.0,11.1],
            [1144.86,1184.54,0.0,11.1],
            [1145.74,1184.6,0.0,11.1],
            [1146.63,1184.66,0.0,11.1],
            [1147.51,1184.72,0.0,11.1],
            [1148.39,1184.78,0.0,11.1],
            [1149.28,1184.84,0.0,11.1],
            [1150.16,1184.89,0.0,11.1],
            [1151.04,1184.95,0.0,11.1],
            [1151.48,1184.98,0.0,11.1],
            [1152.36,1185.03,0.0,11.1],
            [1153.24,1185.09,0.0,11.1],
            [1154.12,1185.14,0.0,11.1],
            [1155.01,1185.19,0.0,11.1],
            [1155.89,1185.24,0.0,11.1],
            [1156.77,1185.29,0.0,11.1],
            [1157.65,1185.34,0.0,11.1],
            [1158.53,1185.39,0.0,11.1],
            [1159.41,1185.44,0.0,11.1],
            [1160.3,1185.49,0.0,11.1],
            [1161.18,1185.54,0.0,11.1],
            [1162.06,1185.59,0.0,11.1],
            [1162.94,1185.64,0.0,11.1],
            [1163.83,1185.69,0.0,11.1],
            [1164.71,1185.73,0.0,11.1],
            [1166.03,1185.81,0.0,11.1],
            [1166.91,1185.86,0.0,11.1],
            [1167.79,1185.9,0.0,11.1],
            [1168.67,1185.95,0.0,11.1],
            [1169.56,1186,0.0,11.1],
            [1170.44,1186.05,0.0,11.1],
            [1171.32,1186.1,0.0,11.1],
            [1172.2,1186.14,0.0,11.1],
            [1173.08,1186.19,0.0,11.1],
            [1173.96,1186.24,0.0,11.1],
            [1175.29,1186.31,0.0,11.1],
            [1176.17,1186.36,0.0,11.1],
            [1177.05,1186.41,0.0,11.1],
            [1177.94,1186.45,0.0,11.1],
            [1178.82,1186.5,0.0,11.1],
            [1179.7,1186.54,0.0,11.1],
            [1180.59,1186.59,0.0,11.1],
            [1181.91,1186.66,0.0,11.1],
            [1182.79,1186.7,0.0,11.1],
            [1183.68,1186.75,0.0,11.1],
            [1184.56,1186.8,0.0,11.1],
            [1185.88,1186.87,0.0,11.1],
            [1186.77,1186.91,0.0,11.1],
            [1187.65,1186.96,0.0,11.1],
            [1188.98,1187.02,0.0,11.1],
            [1189.86,1187.07,0.0,11.1],
            [1190.75,1187.11,0.0,11.1],
            [1192.07,1187.18,0.0,11.1],
            [1192.96,1187.22,0.0,11.1],
            [1193.84,1187.27,0.0,11.1],
            [1195.17,1187.33,0.0,11.1],
            [1196.06,1187.38,0.0,11.1],
            [1196.94,1187.42,0.0,11.1],
            [1198.27,1187.49,0.0,11.1],
            [1199.15,1187.53,0.0,11.1],
            [1200.03,1187.57,0.0,11.1],
            [1201.36,1187.64,0.0,11.1],
            [1202.24,1187.68,0.0,11.1],
            [1203.57,1187.75,0.0,11.1],
            [1204.45,1187.79,0.0,11.1],
            [1205.78,1187.86,0.0,11.1],
            [1206.66,1187.9,0.0,11.1],
            [1207.99,1187.96,0.0,11.1],
            [1208.87,1188,0.0,11.1],
            [1210.19,1188.06,0.0,11.1],
            [1211.08,1188.1,0.0,11.1],
            [1212.4,1188.16,0.0,11.1],
            [1213.29,1188.19,0.0,11.1],
            [1214.62,1188.24,0.0,11.1],
            [1215.5,1188.28,0.0,11.1],
            [1216.83,1188.32,0.0,11.1],
            [1217.71,1188.35,0.0,11.1],
            [1219.04,1188.38,0.0,11.1],
            [1219.92,1188.41,0.0,11.1],
            [1221.24,1188.44,0.0,11.1],
            [1222.13,1188.46,0.0,11.1],
            [1223.45,1188.48,0.0,11.1],
            [1224.34,1188.5,0.0,11.1],
            [1225.66,1188.52,0.0,11.1],
            [1226.55,1188.53,0.0,11.1],
            [1227.87,1188.54,0.0,11.1],
            [1228.76,1188.54,0.0,11.1],
            [1230.09,1188.55,0.0,11.1],
            [1231.41,1188.55,0.0,11.1],
            [1232.29,1188.54,0.0,11.1],
            [1233.62,1188.54,0.0,11.1],
            [1234.5,1188.53,0.0,11.1],
            [1235.83,1188.52,0.0,11.1],
            [1236.71,1188.51,0.0,11.1],
            [1238.04,1188.5,0.0,11.1],
            [1238.92,1188.49,0.0,11.1],
            [1240.25,1188.47,0.0,11.1],
            [1241.13,1188.46,0.0,11.1],
            [1242.46,1188.44,0.0,11.1],
            [1243.34,1188.42,0.0,11.1],
            [1244.67,1188.4,0.0,11.1],
            [1245.55,1188.38,0.0,11.1],
            [1246.88,1188.36,0.0,11.1],
            [1247.76,1188.34,0.0,11.1],
            [1249.09,1188.31,0.0,11.1],
            [1249.97,1188.29,0.0,11.1],
            [1251.29,1188.26,0.0,11.1],
            [1252.18,1188.23,0.0,11.1],
            [1253.5,1188.2,0.0,11.1],
            [1254.39,1188.17,0.0,11.1],
            [1255.72,1188.13,0.0,11.1],
            [1256.6,1188.1,0.0,11.1],
            [1257.92,1188.06,0.0,11.1],
            [1258.81,1188.03,0.0,11.1],
            [1260.13,1187.99,0.0,11.1],
            [1261.02,1187.96,0.0,11.1],
            [1262.34,1187.92,0.0,11.1],
            [1263.23,1187.89,0.0,11.1],
            [1264.56,1187.85,0.0,11.1],
            [1265.44,1187.82,0.0,11.1],
            [1266.76,1187.78,0.0,11.1],
            [1268.09,1187.74,0.0,11.1],
            [1268.97,1187.71,0.0,11.1],
            [1270.3,1187.67,0.0,11.1],
            [1271.18,1187.64,0.0,11.1],
            [1272.51,1187.6,0.0,11.1],
            [1273.83,1187.56,0.0,11.1],
            [1275.16,1187.51,0.0,11.1],
            [1276.04,1187.49,0.0,11.1],
            [1277.37,1187.44,0.0,11.1],
            [1278.69,1187.4,0.0,11.1],
            [1279.58,1187.38,0.0,11.1],
            [1280.9,1187.34,0.0,11.1],
            [1282.23,1187.29,0.0,11.1],
            [1283.12,1187.27,0.0,11.1],
            [1284.44,1187.22,0.0,11.1],
            [1285.76,1187.17,0.0,11.1],
            [1286.65,1187.13,0.0,11.1],
            [1287.97,1187.07,0.0,11.1],
            [1289.3,1187.01,0.0,11.1],
            [1290.18,1186.96,0.0,11.1],
            [1291.5,1186.89,0.0,11.1],
            [1292.38,1186.84,0.0,11.1],
            [1293.71,1186.76,0.0,11.1],
            [1295.03,1186.68,0.0,11.1],
            [1295.91,1186.62,0.0,11.1],
            [1297.23,1186.52,0.0,11.1],
            [1298.11,1186.46,0.0,11.1],
            [1299,1186.4,0.0,11.1],
            [1300.32,1186.3,0.0,11.1],
            [1301.2,1186.23,0.0,11.1],
            [1302.53,1186.12,0.0,11.1],
            [1303.41,1186.05,0.0,11.1],
            [1304.29,1185.98,0.0,11.1],
            [1305.17,1185.9,0.0,11.1],
            [1306.49,1185.79,0.0,11.1],
            [1307.37,1185.71,0.0,11.1],
            [1308.25,1185.63,0.0,11.1],
            [1309.13,1185.55,0.0,11.1],
            [1310.01,1185.46,0.0,11.1],
            [1311.33,1185.34,0.0,11.1],
            [1312.21,1185.25,0.0,11.1],
            [1313.09,1185.16,0.0,11.1],
            [1313.96,1185.08,0.0,11.1],
            [1314.84,1184.98,0.0,11.1],
            [1315.72,1184.89,0.0,11.1],
            [1316.6,1184.81,0.0,11.1],
            [1317.48,1184.71,0.0,11.1],
            [1318.37,1184.62,0.0,11.1],
            [1319.25,1184.53,0.0,11.1],
            [1319.68,1184.48,0.0,11.1],
            [1320.56,1184.39,0.0,11.1],
            [1321.44,1184.3,0.0,11.1],
            [1322.32,1184.21,0.0,11.1],
            [1323.2,1184.12,0.0,11.1],
            [1324.08,1184.03,0.0,11.1],
            [1324.96,1183.93,0.0,11.1],
            [1325.84,1183.85,0.0,11.1],
            [1326.72,1183.76,0.0,11.1],
            [1327.6,1183.66,0.0,11.1],
            [1328.48,1183.58,0.0,11.1],
            [1329.36,1183.48,0.0,11.1],
            [1329.8,1183.44,0.0,11.1],
            [1330.68,1183.35,0.0,11.1],
            [1331.56,1183.26,0.0,11.1],
            [1332.44,1183.17,0.0,11.1],
            [1333.32,1183.08,0.0,11.1],
            [1334.2,1182.99,0.0,11.1],
            [1335.08,1182.9,0.0,11.1],
            [1335.96,1182.81,0.0,11.1],
            [1336.84,1182.72,0.0,11.1],
            [1337.72,1182.64,0.0,11.1],
            [1338.16,1182.59,0.0,11.1],
            [1339.04,1182.51,0.0,11.1],
            [1339.91,1182.42,0.0,11.1],
            [1340.79,1182.33,0.0,11.1],
            [1341.67,1182.25,0.0,11.1],
            [1342.55,1182.16,0.0,11.1],
            [1343.43,1182.08,0.0,11.1],
            [1344.32,1181.99,0.0,11.1],
            [1345.2,1181.91,0.0,11.1],
            [1346.08,1181.82,0.0,11.1],
            [1346.52,1181.78,0.0,11.1],
            [1347.4,1181.7,0.0,11.1],
            [1348.28,1181.61,0.0,11.1],
            [1349.16,1181.53,0.0,11.1],
            [1350.04,1181.44,0.0,11.1],
            [1350.92,1181.36,0.0,11.1],
            [1351.8,1181.28,0.0,11.1],
            [1352.68,1181.19,0.0,11.1],
            [1353.56,1181.11,0.0,11.1],
            [1354,1181.07,0.0,11.1],
            [1354.88,1180.98,0.0,11.1],
            [1355.76,1180.9,0.0,11.1],
            [1356.64,1180.81,0.0,11.1],
            [1357.52,1180.73,0.0,11.1],
            [1358.4,1180.65,0.0,11.1],
            [1359.28,1180.56,0.0,11.1],
            [1359.71,1180.52,0.0,11.1],
            [1360.6,1180.44,0.0,11.1],
            [1361.48,1180.35,0.0,11.1],
            [1362.36,1180.27,0.0,11.1],
            [1363.24,1180.18,0.0,11.1],
            [1364.12,1180.1,0.0,11.1],
            [1364.56,1180.06,0.0,11.1]]



if __name__ == '__main__':
    main()

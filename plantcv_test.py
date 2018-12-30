## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import time
from plantcv import plantcv as pcv
import functions
import csv
import os
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pyplot

# import plotly.plotly as py
# import plotly.graph_objs as go

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

        #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

        #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

        #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

        #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

        #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)  # ->360
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 20 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# default point for distance
target_width = 100
target_height = 100

# Streaming loop
try:
    x640 = np.arange(0, 640)
    y360 = np.arange(0, 360)
    X640, Y360 = np.meshgrid(x640, y360)
    fig = plt.figure()

    # x401 = np.arange(0, 401)
    y401 = np.arange(0, 401)
    X640, Y401 = np.meshgrid(x640, y401)

    loop_i = 0

    while True:
        time.sleep(0.7)
        t1 = time.time()
        loop_i += 1
        if loop_i % 10 == 0:
            cv2.imwrite('ex1.jpg', mask["plant"])
            cv2.imwrite('color_image.jpg', color_image)
            np.savetxt('xyz_all.csv', repre_xyz_all, delimiter=',')  # save CSV
            np.savetxt('xyz.csv', repre_xyz, delimiter=',')  # save CSV
            print("save")
            #np.savetxt('out2.csv', test20, delimiter=',')
            #np.savetxt('out.csv', np_image, delimiter=',')
            #np.savetxt('np_image.csv', np_image, delimiter=',')

        test20 = np.zeros((401, 640))  # 上から見たとき、深度値がどれくらい重なっているかを記録 初期値0　400mmまで記録

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        s = pcv.rgb2gray_hsv(color_image, 's')  # plantcv
        s_thresh = pcv.threshold.binary(s, 85, 255, 'light')  # plantcv
        s_mblur = pcv.median_blur(s_thresh, 5)
        s_cnt = pcv.median_blur(s_thresh, 5)
        # cv2.imshow('color - depth3', s_mblur)
        # Convert RGB to LAB and extract the Blue channel
        b = pcv.rgb2gray_lab(color_image, 'b')

        # Threshold the blue image
        b_thresh = pcv.threshold.binary(b, 160, 255, 'light')
        b_cnt = pcv.threshold.binary(b, 160, 255, 'light')

        # Fill small objects
        # b_fill = pcv.fill(b_thresh, 10)
        mask = pcv.naive_bayes_classifier(color_image, "naive_bayes_pdfs.txt")
        #histogram_low = np.sum(mask["plant"][int(360/2):, :], axis=0)
        #histogram_up = np.sum(mask["plant"][:int(360/2), :], axis=0)
        histogram = np.sum(mask["plant"][int(360/2):, :], axis=0)
        win_left_x = np.argmax(histogram)
        branch_x, branch_y, left_dot_x, left_dot_y = functions.sliding_windows(mask["plant"], win_left_x)
        # cv2.circle(mask["plant"], (np.argmax(histogram_low), 400), 30, (255, 0, 0), -1)
        # cv2.circle(mask["plant"], (np.argmax(histogram_up), 200), 30, (255, 0, 0), -1)
        cv2.imshow('color - depthmask', mask["plant"])
        # np.savetxt('histogram.csv', histogram,delimiter=',')
        #w = csv.writer(open("output.csv", "w"))
        #for key, val in mask.items():
                #w.writerow([key, val])




        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # print(depth_image_3d)
        # print(type(depth_image_3d)) # <class 'numpy.ndarray'>

        # depth1 = frames.get_distance(300, 150)
        meters = frames.get_depth_frame()
        meters2 = meters.get_distance(target_width, target_height)
        # print("depth", depth_image_3d[300][150])
        # print("depth", meters2)  #show point distance

        # depthデータをCSV出力する
        depth = aligned_frames.get_depth_frame()
        depth_data = depth.as_frame().get_data()
        np_image = np.asanyarray(depth_data)

        vertical_0 = 150
        vertical_100 = 210
        horizontal_0 = 270
        horizontal_100 = 370  # test20 = 401*640

        for i in range(vertical_0, vertical_100):
            for j in range(horizontal_0, horizontal_100):
                # for j in range(10, 630):
                if np_image[i][j] <= 400 and np_image[i][j] >= 100:  # 400mmまでに茎がないかどうか
                    test20[np_image[i][j]][j] += 1
        vertical_0 -= 15
        vertical_100 += 15

        branch_x_array = np.array(branch_x)
        branch_y_array = np.array(branch_y)
        left_dot_x_array = np.array(left_dot_x)
        left_dot_y_array = np.array(left_dot_y)
        v_vertex_x = np.zeros(len(branch_x))
        v_vertex_y = np.zeros(len(branch_y))
        v_vertex_z = np.zeros(len(branch_x))
        v_vertex_x_all = np.zeros(len(left_dot_x))
        v_vertex_y_all = np.zeros(len(left_dot_x))
        v_vertex_z_all = np.zeros(len(left_dot_x))
        # 3d位置を計算

        if len(branch_x) != 0:
            for points_i in range(len(branch_x)):
                    depth_for_rep_zero = (np_image[branch_y_array[points_i]][branch_x_array[points_i]+1],
                                        np_image[branch_y_array[points_i]][branch_x_array[points_i]+2],
                                        np_image[branch_y_array[points_i]][branch_x_array[points_i]-1],
                                        np_image[branch_y_array[points_i]][branch_x_array[points_i]-2],)
                    depth_for_rep = min([e for e in depth_for_rep_zero if depth_for_rep_zero != 0])
                    v_vertex_x[points_i] = branch_x_array[points_i] * depth_for_rep / 3.48
                    v_vertex_y[points_i] = branch_y_array[points_i] * depth_for_rep / 3.48
                    v_vertex_z[points_i] = depth_for_rep
        if len(left_dot_x) != 0:
            for points_i in range(len(left_dot_x)):
                depth_for_rep_zero = (np_image[left_dot_y_array[points_i]][left_dot_x_array[points_i]+1],
                                      np_image[left_dot_y_array[points_i]][left_dot_x_array[points_i]+2],
                                      np_image[left_dot_y_array[points_i]][left_dot_x_array[points_i]-1],
                                      np_image[left_dot_y_array[points_i]][left_dot_x_array[points_i]-2],)
                depth_for_rep = min([e for e in depth_for_rep_zero if depth_for_rep_zero != 0])
                v_vertex_x_all[points_i] = left_dot_x_array[points_i] * depth_for_rep / 3.48
                v_vertex_y_all[points_i] = left_dot_y_array[points_i] * depth_for_rep / 3.48
                v_vertex_z_all[points_i] = depth_for_rep

        repre_xyz = np.hstack((v_vertex_x, v_vertex_y, v_vertex_z, len(v_vertex_z)))
        repre_xyz_all = np.hstack((v_vertex_x_all, v_vertex_y_all, v_vertex_z_all, len(v_vertex_z_all)))
        # print(len(v_vertex_z))

        # np.savetxt('xyzaa.csv', np_image, delimiter=',')  # save CSV

        ## print("range ", vertical_0, vertical_100)
        ## print(np_image[180][320])
        res1 = np.argmax(test20)
        res4 = np.amax(test20)
        res2, res3 = divmod(res1, 640)
        res5 = test20[res2, res3]
        # print(res1, res2, res3, res4, res5)
        # print(test20)
        test40 = np.where(np_image[:, [res3]] == res2)
        # print(test40[0][:])
        images = color_image

        if res3 != 0:
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            ave_hsv = np.zeros(3)
            ave_hsv_params = np.zeros(3)
            for res3_i in range(0, int(res4)):  # 特定の垂線のみ探索
                if np_image[test40[0][res3_i], res3+2] >= res2-1 and np_image[test40[0][res3_i], int(res4)+3] <= res2+1:
                    ave_hsv = hsv[test40[0][res3_i], res3+2]
                    cv2.drawMarker(images, (res3+2, test40[0][res3_i]), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=15)
                elif np_image[test40[0][res3_i], res3-2] >= res2-1 and np_image[test40[0][res3_i], int(res4)+3] <= res2+1:
                    ave_hsv = hsv[test40[0][res3_i], res3-2]
                    cv2.drawMarker(images, (res3-2, test40[0][res3_i]), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=15)
                for res3_j in range(0, 3):
                    ave_hsv_params[res3_j] += ave_hsv[res3_j]
            ave_h = ave_hsv_params[0] / int(res4)
            ave_s = ave_hsv_params[1] / int(res4)
            ave_v = ave_hsv_params[2] / int(res4)
            # print("hsv", ave_h, ave_s, ave_v)

        '''# x_rep = [res3, res3]
        # y_rep = [10, 350]
        # z_rep = [res2, res2]
        # ax = Axes3D(fig)
        # ax.plot(x_rep, z_rep, y_rep, "o-", color="#00aa00", ms=4, mew=0.5)
        plt.xlim(0, 640)
        plt.ylim(0, 400)
        plt.xlabel("x axis")
        plt.ylabel("z axis")
        plt.plot(res3, res2, marker='.', markersize=20)
        plt.pause(0.5)
        plt.clf()'''

        # pyplot.plot(test20)
        # pyplot.show()

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2GRAY)
        # kernel = np.array([[0, 0, 0],
                           # [-1, 0, 1],
                           # [0, 0, 0]])
        # dst1 = cv2.Canny(color_image, 75, 150)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        ## center_param = hsv[target_width][target_height]
        ## print(center_param)
        if res3 != 0:
            lower_yellow = np.array([ave_h, ave_s-40, ave_v-40])
            upper_yellow = np.array([ave_h+80, ave_s+10, ave_v])
            img_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            dst1 = cv2.bitwise_and(hsv, hsv, mask=img_mask)
            # cv2.imshow('Masked', dst1)

        # height = dst1.shape[0]
        # width = dst1.shape[1]
        # print("", width, height)


        # images = depth_colormap
        # images = np.hstack((color_image, depth_colormap))
        # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        # cv2.circle(images, (target_width, target_height), 10, (255, 0, 0), 5)
        # cv2.line(images, (res3, 0), (res3, 480), (255, 0, 0), 5)
        # cv2.imshow('color - depth', images)
        # cv2.imshow('color - depth2', hsv)  # RGB+depth(Colored)
        # cv2.imshow('Align Example2', np_image)  # ndarray data
        # cv2.circle(dst1, (target_width, target_height), 10, (255, 0, 0), -1)


        # print(type(np_image))

        '''# 配列を3Dグラフで表示
        ax = Axes3D(fig)
        ax.plot_wireframe(X640, Y401, test20)
        # plt.tight_layout()
        plt.pause(1.5)
        print("plot")'''

        mouseData = mouseParam('color - depth')

        cv2.waitKey(20)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            target_width = mouseData.getX()
            target_height = mouseData.getY()
            print("mouse_point", target_width, target_height)
            # np.savetxt('out.csv', np_image, delimiter=',') # save CSV
        #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break;

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        t2 = time.time()
        elapsed_time = t2-t1
        # print(f"経過時間：{elapsed_time}")
finally:

    pipeline.stop()


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
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
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
        t1 = time.time()
        loop_i += 1
        if loop_i % 1000 == 0:
            np.savetxt('out2.csv', test20, delimiter=',')
            np.savetxt('out.csv', np_image, delimiter=',')
            cv2.imwrite('ex1.jpg', images)

        test20 = np.zeros((401, 640))  # 上から見たとき、深度値がどれくらい重なっているかを記録 初期値0　400mmまで記録

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

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
        depth = frames.get_depth_frame()
        depth_data = depth.as_frame().get_data()
        np_image = np.asanyarray(depth_data)
        # print(test20)

        for i in range(10, 350):
            for j in range(270, 370):
            # for j in range(10, 630):
                if np_image[i][j] <= 400 and np_image[i][j] >= 100:  # 400mmまでに茎がないかどうか
                    test20[np_image[i][j]][j] += 1
        print(np_image[180][320])
        res1 = np.argmax(test20)
        res4 = np.amax(test20)
        res2, res3 = divmod(res1, 640)
        print(res1, res2, res3, res4)
        # print(test20)
        test40 = np.where(np_image[:, [res3]] == res2)
        print(test40[0][:])

        # x_rep = [res3, res3]
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
        plt.clf()

        # pyplot.plot(test20)
        # pyplot.show()

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2GRAY)
        kernel = np.array([[0, 0, 0],
                           [-1, 0, 1],
                           [0, 0, 0]])
        dst1 = cv2.Canny(gray, 75, 150)
        height = dst1.shape[0]
        width = dst1.shape[1]
        # print("", width, height)

        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.circle(images, (target_width, target_height), 10, (255, 0, 0), 5)
        cv2.imshow('Align Example', images)  # RGB+depth(Colored)
        # cv2.imshow('Align Example2', np_image)  # ndarray data
        cv2.circle(dst1, (target_width, target_height), 10, (255, 0, 0), -1)
        # cv2.imshow('Align Example3', dst1)  # cv2.canny

        # print(type(np_image))

        '''# 配列を3Dグラフで表示
        ax = Axes3D(fig)
        ax.plot_wireframe(X640, Y401, test20)
        # plt.tight_layout()
        plt.pause(1.5)
        print("plot")'''

        mouseData = mouseParam('Align Example')

        cv2.waitKey(20)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            target_width = mouseData.getX()
            target_height = mouseData.getY()
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
        print(f"経過時間：{elapsed_time}")
finally:

    pipeline.stop()


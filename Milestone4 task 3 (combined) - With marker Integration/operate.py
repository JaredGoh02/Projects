# teleoperate the robot, perform SLAM and object detection

# basic python packages
from operator import indexOf
from tkinter import Image
import numpy as np
import cv2 
import os, sys
import time
import json
import ast

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

#Import Task3 Items
from pathlib import Path
# import cv2
import math
from machinevisiontoolbox import Image
import statistics

import matplotlib.pyplot as plt
import PIL

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.mtx = np.array([[1.06374294e+3, 0, 3.52330611e+02],[0,1.06561853e+03,2.57858244e+02],[0,0,1]])
        self.dist = np.array([-4.24422461e-01,-5.78906352e-01,-9.31874026e-04,4.24078376e-05,5.26953417e+00])
        ### Task1 ###
        self.order =0
        self.Complete_t_d =True
        self.lv, self.rv = [0,0]
        self.drive_time =[0]
        self.turn_time =[0]
        self.turntime_left_around =None
        self.turntime_left = None
        self.drivetime_left = None
        ### Task2 ###
        self.fruits_true_pos = []
        self.aruco_true_pos=[]
        self.start = False
        self.around_order =0
        self.interlock_turn_around = True
        self.points =[]
        self.TruePositions = []
        ### Task3 ###
        self.fruits_list = []
        self.search_list = []
        self.numFruitList = []
        self.FruitsToNum_dict = {"redapple": 1, "greenapple": 2, "orange": 3, "mango": 4, "capsicum": 5}
        self.NumToFruits_dict = {v: k for k, v in self.FruitsToNum_dict.items()}
        self.camera_matrix = []
        self.RegenComplete = False
        self.RegenerateFlag = False

        self.RedapplePics = []
        self.GreenapplePics = []
        self.OrangePics = []
        self.MangoPics = []
        self.CapsicumPics = []


        self.RedapplePose = []
        self.GreenapplePose = []
        self.OrangePose = []
        self.MangoPose = []
        self.CapsicumPose = []



    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            # self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.yolo_detection(self.img)###########yolov5
            #self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)#####resnet
            print(self.detector_output)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'


    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # if self.start:
            #     start = True
            # else: 
            #     start = False
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2.5, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2.5, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 3]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -3]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                self.start = True
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    # if not self.ekf_on:
                        # self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                #self.command['inference'] = True
                self.RegenerateMap()
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


    ############ Milestone 4 ###################
    def read_true_map(self, fname):
            """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
            @param fname: filename of the map
            @return:
                1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
                2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
                3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
            """
            with open(fname, 'r') as f:
                try:
                    gt_dict = json.load(f)                   
                except ValueError as e:
                    with open(fname, 'r') as f:
                        gt_dict = ast.literal_eval(f.readline())   
                fruit_list = []
                fruit_true_pos = []
                aruco_true_pos = np.empty([10, 2])

                # remove unique id of targets of the same type
                for key in gt_dict:
                    x = np.round(gt_dict[key]['x'], 1)
                    y = np.round(gt_dict[key]['y'], 1)

                    if key.startswith('aruco'):
                        if key.startswith('aruco10'):
                            aruco_true_pos[9][0] = x
                            aruco_true_pos[9][1] = y
                        else:
                            marker_id = int(key[5])
                            # print(marker_id)
                            aruco_true_pos[marker_id-1][0] = x
                            aruco_true_pos[marker_id-1][1] = y
                    else:
                        fruit_list.append(key[:-2])
                        if len(fruit_true_pos) == 0:
                            fruit_true_pos = np.array([[x, y]])
                        else:
                            fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

                return fruit_list, fruit_true_pos, aruco_true_pos

    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list

    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """
        self.fruit_orderlist = []
        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(3):
                if fruit == fruit_list[i]:
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                      fruit,
                                                      np.round(fruit_true_pos[i][0], 1),
                                                      np.round(fruit_true_pos[i][1], 1)))
                    if len(self.fruit_orderlist) == 0:
                        self.fruit_orderlist = np.array([[fruit,fruit_true_pos[i][0],fruit_true_pos[i][1]]])
                    else:
                        self.fruit_orderlist = np.append(self.fruit_orderlist, [[fruit,fruit_true_pos[i][0],fruit_true_pos[i][1]]], axis=0)
            n_fruit += 1

    def get_robot_pose(self):
        return self.ekf.get_state_vector() 

    ################## Task 1 #############################
    # wheel control

    def innitMarkers(self, Positions):
        for i in range(len(Positions)):
            x = Positions[i][0]
            y = Positions[i][1]
            marker = np.block([[x],[y]]).reshape(-1,1)

            landmarks = measure.Marker(marker, i+1)
            self.TruePositions.append(landmarks)
        self.ekf.add_landmarks(self.TruePositions)




    def control(self, waypoint, robot_pose):       
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',') #Distance ticks/m
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',') #angle ticks/rad

        wheel_lv = 30 # tick to move the robot
        wheel_lr = 30
        
        
        
        if args.play_data:
            self.lv, self.rv = self.pibot.set_velocity()            
        else:
            x_goal, y_goal = waypoint
            x, y, __ = [robot_pose[0],robot_pose[1],robot_pose[2]]
            print('robot pose')
            print(robot_pose)

            if robot_pose[2]>=(np.pi*2):
                theta = robot_pose[2]%(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta -np.pi*2
            elif robot_pose[2]<(-np.pi*2):
                theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta-np.pi*2
            elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
                theta = robot_pose[2] + (np.pi*2)
            elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
                theta = robot_pose[2] - (np.pi*2)
            else:
                theta = robot_pose[2]
            # print(theta)

            # if x_goal >= x:
            x_diff = np.round_((x_goal - x),8)
            # else:
            #     x_diff = x - x_goal

            # if y_goal >= y:
            y_diff = np.round_((y_goal - y),8)
            # else:
                # y_diff = y - y_goal

            #Obtain angle difference
            alpha = np.arctan2(y_diff, x_diff) - theta

            if alpha>np.pi:
                alpha = alpha - 2*np.pi
            elif alpha < -np.pi:
                alpha = alpha + 2*np.pi
            print(np.arctan2(y_diff, x_diff))
            print(theta)
            print(alpha)
            # print("theta")
            # print(y_diff)
            # print(theta)
            # print(np.arctan2(y_diff, x_diff))
            # print(alpha)
        
            #Obtain length difference
            rho = np.clip(np.hypot(x_diff, y_diff),0,0.6)

            self.turn_time = abs(alpha*(baseline*np.pi/scale/wheel_lr)/(2*np.pi)) # replace with your calculation
            # self.turn_time = (-0.01485446135524356467176248458143*(abs(alpha)-np.pi/4)+1)*self.turn_time
            self.turn_time = (-0.118732395447351627*(abs(alpha)-np.pi/4)+1)*self.turn_time
            self.drive_time = rho * (   (1/scale)/(wheel_lv)   )  # replace with your calculation
            nextwaypoint_bool = False
            if rho < 0.1:
                nextwaypoint_bool = True
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
                self.order =0
                self.Complete_t_d =True
                print('next waypoint')
                time.sleep(2)
            elif self.order ==0 and self.Complete_t_d ==True and nextwaypoint_bool == False: ##turning
                print("Turning for {:.5f} seconds".format(self.turn_time[0]))
                # self.turn_time =0
                if alpha >0:
                    self.lv, self.rv = self.pibot.set_velocity([0, 1], tick=20,turning_tick=wheel_lr, time= self.turn_time[0])
                    self.lv = (0.118732395447351627*(abs(alpha)-np.pi/4)+1)*self.lv*1.084
                    self.rv = (0.118732395447351627*(abs(alpha)-np.pi/4)+1)*self.rv*1.084
                    # print(self.lv, self.rv)
                elif alpha <0:
                    self.lv, self.rv = self.pibot.set_velocity([0, -1], tick=20,turning_tick=wheel_lr,time= self.turn_time[0])
                    self.lv = (0.118732395447351627*(abs(alpha)-np.pi/4)+1)*self.lv*1.084
                    self.rv = (0.118732395447351627*(abs(alpha)-np.pi/4)+1)*self.rv*1.084
                else:
                    self.lv, self.rv = self.pibot.set_velocity([0, 0])
                self.order =1
                self.drivetime_left = None
                self.Complete_t_d =False

                #Keep Track of when code has been executed
                # self.Time_Of_execution_t = time.time()
            # elif self.order ==1 and (time.time() - self.Time_Of_execution_t > self.turn_time[0]) and nextwaypoint_bool == False: ## driving
            elif self.order ==1 and nextwaypoint_bool == False: ## driving
                print("Driving for {:.5f} seconds".format(self.drive_time[0]))
                self.lv, self.rv = self.pibot.set_velocity([1, 0], tick=wheel_lv, time=self.drive_time[0])
                self.order =0
                self.turntime_left = None
                # self.Time_Of_execution_d = time.time()
            # elif self.order ==0 and (time.time()-self.Time_Of_execution_d > self.drive_time[0])and nextwaypoint_bool == False:
            elif self.order ==0 and  nextwaypoint_bool == False:
                self.Complete_t_d =True
                self.lv, self.rv = self.pibot.set_velocity([0, 0])


        time.sleep(1.5)
        ####test
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1.5
        if self.order==0:
            if self.drive_time[0] > dt:
                if self.drivetime_left==None:
                    self.drivetime_left = self.drive_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.drivetime_left<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.drivetime_left)
                    self.drivetime_left ==None
                elif self.drivetime_left!=None:
                    self.drivetime_left = self.drivetime_left-dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
            else:
                drive_meas = measure.Drive(self.lv, self.rv, self.drive_time[0])
        else:
            if self.turn_time[0] > dt:
                if self.turntime_left==None:
                    self.turntime_left = self.turn_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.turntime_left<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left)
                    self.turntime_left ==None
                elif self.turntime_left!=None:
                    self.turntime_left = self.turntime_left-dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
            else:
                drive_meas = measure.Drive(self.lv, self.rv, self.turn_time[0])
        
        # drive_meas = measure.Drive(self.lv, self.rv, dt)
        self.control_clock = time.time()
        operate.update_slam(drive_meas)
            
        return nextwaypoint_bool
    def turn_around(self, robot_pose):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',') #Distance ticks/m
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',') #angle ticks/rad
        # baseline = 1.350086542959292168e-01
        # baseline = 1.300086542959292168e-01
        # wheel_lv = 30 # tick to move the robot
        wheel_lr = 30
        print(robot_pose)

        if robot_pose[2]>=(np.pi*2):
            theta = robot_pose[2]%(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta -np.pi*2
        elif robot_pose[2]<(-np.pi*2):
            theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta-np.pi*2
        elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
            theta = robot_pose[2] + (np.pi*2)
        elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
            theta = robot_pose[2] - (np.pi*2)
        else:
            theta = robot_pose[2]

        if self.around_order ==0 and self.interlock_turn_around == True:
            self.interlock_turn_around = False
            theta_first = theta
            self.points = [theta_first+np.pi/4,theta_first+np.pi/2,theta_first+np.pi*3/4,theta_first+np.pi,theta_first-np.pi*3/4,theta_first-np.pi/2,theta_first-np.pi/4,theta_first]
        
        alpha = self.points[self.around_order] - theta

        if alpha>np.pi:
            alpha = alpha - 2*np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2*np.pi

        self.turn_time_around = abs(alpha*(baseline*np.pi/scale/wheel_lr)/(2*np.pi)) 
        done_ =False
        if abs(alpha)<(5/180*np.pi):
            self.around_order +=1
            self.turntime_left_around==None
            self.lv, self.rv = self.pibot.set_velocity([0, 0])
            
            self.RegenerateMap()

            if self.around_order >=8:
                self.interlock_turn_around = True
                self.around_order =0
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
                time.sleep(1)
                done_ =True
                print("DONEEEE")
            # print('wwwww')
            # print(self.around_order)
        else:
            print("Turning for {:.5f} seconds".format(self.turn_time_around[0]))
            # self.turn_time =0
            if alpha >0:
                self.lv, self.rv = self.pibot.set_velocity([0, 1], tick=20,turning_tick=wheel_lr, time= self.turn_time_around[0])
                # print(self.lv, self.rv)
            elif alpha <0:
                self.lv, self.rv = self.pibot.set_velocity([0, -1], tick=20,turning_tick=wheel_lr,time= self.turn_time_around[0])
            else:
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
        # print('dddd')
        # print(theta) 
        # print(alpha)
        # # print(self.points[self.around_order])
        # # print(5/180*np.pi)
        # print(done_)
        
        time.sleep(1.5)
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1.5

        if self.turn_time_around[0] > dt:
            if self.turntime_left_around==None:
                self.turntime_left_around = self.turn_time_around[0] - dt
                drive_meas = measure.Drive(self.lv, self.rv, dt)
            elif self.turntime_left_around<dt:
                drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left_around)
                self.turntime_left_around ==None
            elif self.turntime_left_around!=None:
                self.turntime_left_around = self.turntime_left_around-dt
                drive_meas = measure.Drive(self.lv, self.rv, dt)
        else:
            drive_meas = measure.Drive(self.lv, self.rv, self.turn_time_around[0])
        
        # drive_meas = measure.Drive(self.lv, self.rv, dt)
        self.control_clock = time.time()
        operate.update_slam(drive_meas)
            
        return done_
    ################Task 2########################
    def obstacles_detector(self,node):##return True if obstacles is near (added by zi yu)
        for count,(x,y) in enumerate(self.aruco_true_pos):
            if count!=0:
                x_diff = node[0] - x
                y_diff = node[1] - y
                if np.hypot(x_diff, y_diff)<0.15:
                    return True

        for x,y in self.fruits_true_pos:
            x_diff = node[0] - x
            y_diff = node[1] - y
            if np.hypot(x_diff, y_diff)<0.15:
                return True
        return False
    
    def displacement_from_target(self,node,target): ##return dispalcement of target and node (added by zi yu)
        x_diff = node[0] - float(target[1])
        y_diff = node[1] -float(target[2])
        return np.hypot(x_diff, y_diff)

    def neighbor_nodes(self,robot_posnode):
        
        nodes = []
        for i in range(9):
            if i ==0:
                x____ = robot_posnode[0] -0.2
                y____ = robot_posnode[1] +0.2
                nodes = np.array([[x____,y____]])
            else:
                x____ += 0.2
                x____ = np.round_(x____, decimals=1)
                if (i == 3 or i ==6) and i!=0:
                    y____ -= 0.2
                    y____ = np.round_(y____, decimals=1)
                    x____ = robot_posnode[0] -0.2
                if i!=4:
                    if len(nodes)==0:
                        nodes = np.array([[x____,y____]])
                    else:
                        nodes = np.append(nodes,[[x____,y____]],axis=0)
        return nodes

    def optimum_nodes_to_target(self,neighbor_nodes,target):
        optimum_nodes = []
        dis = 999999
        index_action =9
        for index,node in enumerate(neighbor_nodes):
            dis_node_to_target = self.displacement_from_target(node,target)
            if (index == 0 or index ==2 or index ==5 or index ==7) and (dis>(dis_node_to_target+0.09)) and not (self.obstacles_detector(node)):
                optimum_nodes = node
                dis = dis_node_to_target
                index_action =index
            elif not (index == 0 or index ==2 or index ==5 or index ==7) and dis>dis_node_to_target and not (self.obstacles_detector(node)):
                optimum_nodes = node
                dis = dis_node_to_target
                index_action =index
        return optimum_nodes,index_action

    def optimum_waypoint_from_robotpos(self,robot_posnode,target): 
        completesearch_waypoint = False
        # previous_pos = np.reshape(robot_posnode, (-1))
        neighbor_nodes = self.neighbor_nodes(robot_posnode)
        neighbor_nodes = np.reshape(neighbor_nodes, (-1, 2))
        optimum_nodes,index = self.optimum_nodes_to_target(neighbor_nodes,target)
        optimum_nodes = np.reshape(optimum_nodes, (-1, 2))
        waypoints =np.array(optimum_nodes)
        index_action =np.array([[index]])

        while not (completesearch_waypoint):
            neighbor_nodes = self.neighbor_nodes(waypoints[-1,:])
            neighbor_nodes = np.reshape(neighbor_nodes, (-1, 2))
            optimum_nodes,index = self.optimum_nodes_to_target(neighbor_nodes,target)
            optimum_nodes = np.reshape(optimum_nodes, (-1, 2))

            # if index_action[-1] ==index or np.dot([optimum_nodes[-1,0]-waypoints[-1,0],optimum_nodes[-1,1]-waypoints[-1,1]],[waypoints[-1,0]-previous_pos[0],waypoints[-1,1]-previous_pos[1]])==0:
            if index_action[-1] ==index:
                waypoints[-1]=optimum_nodes
                index_action[-1] = index
            else:
                waypoints = np.append(waypoints,optimum_nodes,axis=0)
                index_action = np.append(index_action,[[index]],axis=0)

            # if len(waypoints)>1:
            #     previous_pos = np.reshape(waypoints[-2], (-1))
                
            if self.displacement_from_target(waypoints[-1,:],target)<0.4:
                completesearch_waypoint = True
        return waypoints

        ########## Task 3 ##########
    def get_bounding_box(self, target_number, img):
        image = img.resize((640,480), PIL.Image.Resampling.NEAREST)
        target = Image(img)==target_number
        print(target)
        blobs = target.blobs()
        [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
        width = abs(u1-u2)
        height = abs(v1-v2)
        center = np.array(blobs[0].centroid).reshape(2,)
        box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
        # plt.imshow(fruit.image)
        # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
        # plt.show()
        # assert len(blobs) == 1, "An image should contain only one object of each target type"
        return box

    # read in the list of detection results with bounding boxes and their matching robot pose info
    def get_image_info(self, img, image_poses):
        # there are at most five types of targets in each image
        target_lst_box = [[], [], [], [], []]
        target_lst_pose = [[], [], [], [], []]
        completed_img_dict = {}

        # add the bounding box info of each target in each image
        # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target

        
        img_vals = set(Image(img, grey=True).image.reshape(-1))
        # print('image values')
        # print(img_vals)
        for target_num in img_vals:
            if target_num > 0:
                try:
                    box = self.get_bounding_box(target_num, img) # [x,y,width,height]
                    pose = [image_poses[0], image_poses[1], image_poses[2]] # [x, y, theta]
                    target_lst_box[int(target_num-1)].append(box) # bouncing box of target
                    target_lst_pose[int(target_num-1)].append(np.array(pose).reshape(3,)) # robot pose
                except ZeroDivisionError:
                    pass

        # if there are more than one objects of the same type, combine them
        for i in range(5):
            if len(target_lst_box[i])>0:
                box = np.stack(target_lst_box[i], axis=1)
                pose = np.stack(target_lst_pose[i], axis=1)
                completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
        return completed_img_dict

    # estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
    def estimate_pose(self, camera_matrix, completed_img_dict):
        camera_matrix = camera_matrix
        focal_length = camera_matrix[0][0]
        # actual sizes of targets [For the simulation models]
        # You need to replace these values for the real world objects
        target_dimensions = []
        redapple_dimensions = [0.074, 0.074, 0.087]
        target_dimensions.append(redapple_dimensions)
        greenapple_dimensions = [0.081, 0.081, 0.067]
        target_dimensions.append(greenapple_dimensions)
        orange_dimensions = [0.075, 0.075, 0.072]
        target_dimensions.append(orange_dimensions)
        mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
        target_dimensions.append(mango_dimensions)
        capsicum_dimensions = [0.073, 0.073, 0.088]
        target_dimensions.append(capsicum_dimensions)

        target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

        target_pose_dict = {}
        # for each target in each detection output, estimate its pose
        for target_num in completed_img_dict.keys():
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num-1][2]
            
            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            # This is the default code which estimates every pose to be (0,0)
            # print('theta')
            # if robot_pose[2]>=(np.pi*2):
            #     robot_pose__ = robot_pose[2]%(np.pi*2)
            # elif robot_pose[2]<(-np.pi*2):
            #     robot_pose__ = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
            # elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<0:
            #     robot_pose__ = robot_pose[2] + (np.pi*2)
            # else:
            robot_pose__ = robot_pose[2]

            # print(box)
            z = true_height*focal_length/box[3]
            x_pose = np.sin(robot_pose__)* (z*(640/2-box[0])/focal_length) + np.cos(robot_pose__)* z + robot_pose[0]
            y_pose = np.cos(robot_pose__)* (z*-(640/2-box[0])/focal_length) - np.sin(robot_pose__)* z + robot_pose[1]

            target_pose = {'x': x_pose, 'y': y_pose}
            
            target_pose_dict[target_list[target_num-1]] = target_pose
            ###########################################
        
        return target_pose_dict
   
    def merge_estimations(target_map):
        target_map = target_map
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
        target_est = {}
        num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
        # combine the estimations from multiple detector outputs
        for f in target_map:
            for key in target_map[f]:
                if key.startswith('redapple'):
                    redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('greenapple'):
                    greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('orange'):
                    orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('mango'):
                    mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('capsicum'):
                    capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

        ######### Replace with your codes #########
        # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
        # Replace it with a better merge solution.
        # if len(redapple_est) > num_per_target:
        # print(redapple_est)
        # i = int(input("redapple_est index"))
        redapple_est_x = statistics.median(np.reshape(redapple_est,(-1,2))[:,0])
        redapple_est_y = statistics.median(np.reshape(redapple_est,(-1,2))[:,1])
        # if len(greenapple_est) > num_per_target:
        # print(greenapple_est)
        # i = int(input("greenapple_est index"))
        greenapple_est_x = statistics.median(np.reshape(greenapple_est,(-1,2))[:,0])
        greenapple_est_y = statistics.median(np.reshape(greenapple_est,(-1,2))[:,1])
        # if len(orange_est) > num_per_target:
        # print(orange_est)
        # i = int(input("orange_est index"))
        orange_est_x = statistics.median(np.reshape(orange_est,(-1,2))[:,0])
        orange_est_y = statistics.median(np.reshape(orange_est,(-1,2))[:,1])
        # if len(mango_est) > num_per_target:
        # print(mango_est)
        # i = int(input("mango_est index"))
        mango_est_x = statistics.median(np.reshape(mango_est,(-1,2))[:,0])
        mango_est_y = statistics.median(np.reshape(mango_est,(-1,2))[:,1])
        # if len(capsicum_est) > num_per_target:
        # print(capsicum_est)
        # i = int(input("capsicum_est index"))
        capsicum_est_x = statistics.median(np.reshape(capsicum_est,(-1,2))[:,0])
        capsicum_est_y = statistics.median(np.reshape(capsicum_est,(-1,2))[:,1])

        for i in range(num_per_target):
            try:
                target_est['redapple_'+str(i)] = {'x':redapple_est_x, 'y':redapple_est_y}
            except:
                pass
            try:
                target_est['greenapple_'+str(i)] = {'x':greenapple_est_x, 'y':greenapple_est_y}
            except:
                pass
            try:
                target_est['orange_'+str(i)] = {'x':orange_est_x, 'y':orange_est_y}
            except:
                pass
            try:
                target_est['mango_'+str(i)] = {'x':mango_est_x, 'y':mango_est_y}
            except:
                pass
            try:
                target_est['capsicum_'+ str(i)] = {'x':capsicum_est_x, 'y':capsicum_est_y}
            except:
                pass
        ###########################################
        
        return target_est

    def RegenerateMap(self):
            # Take picture of surroundings
            # Find fruits within it
            # If fruit is found that is not on the list execute the following
            # Else Continue


       
        self.command['inference'] = True
        self.take_pic()
        self.detect_target()
        DiscoveredItems = np.unique(self.detector_output)
        PotentialObstacle = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        currentPose = self.get_robot_pose()

        for i in range(len(DiscoveredItems)):
            if self.numFruitList.count(DiscoveredItems[i]) == 0 and not(DiscoveredItems[i] == 0):

                if DiscoveredItems[i] == 1:
                    self.RedapplePics.append([self.img])
                    self.RedapplePose.append([currentPose[0], currentPose[1], currentPose[2]])

                elif DiscoveredItems[i] == 2:
                    self.GreenapplePics.append([self.img])
                    self.GreenapplePose.append([currentPose[0], currentPose[1], currentPose[2]])

                elif DiscoveredItems[i] == 3:
                    self.OrangePics.append([self.img])
                    self.OrangePose.append([currentPose[0], currentPose[1], currentPose[2]])

                elif DiscoveredItems[i] == 4:
                    self.MangoPics.append([self.img])
                    self.MangoPose.append([currentPose[0], currentPose[1], currentPose[2]])

                elif DiscoveredItems[i] == 5:
                    self.CapsicumPics.append([self.img])
                    self.CapsicumPose.append([currentPose[0], currentPose[1], currentPose[2]])

        
        
        if(self.fruits_list.count(PotentialObstacle[0]) == 0 and len(self.RedapplePics)) >= 3:
            self.numFruitList.append(int(i+1))
            self.fruits_list.append(PotentialObstacle[int(i)])
            self.EstimationAndMerge(self.RedapplePics, self.RedapplePose)
        
        if(self.fruits_list.count(PotentialObstacle[1]) == 0 and len(self.GreenapplePics)) >= 3:
            self.numFruitList.append(int(i+1))
            self.fruits_list.append(PotentialObstacle[int(i)])
            self.EstimationAndMerge(self.GreenapplePics, self.GreenapplePics)

        if(self.fruits_list.count(PotentialObstacle[2]) == 0 and len(self.OrangePics) >= 3):
            self.numFruitList.append(int(i+1))
            self.fruits_list.append(PotentialObstacle[int(i)])
            self.EstimationAndMerge(self.OrangePics, self.OrangePics)

        if(self.fruits_list.count(PotentialObstacle[3]) == 0 and len(self.MangoPics) >= 3):
            self.numFruitList.append(int(i+1))
            self.fruits_list.append(PotentialObstacle[int(i)])
            self.EstimationAndMerge(self.MangoPics, self.MangoPics)

        if(self.fruits_list.count(PotentialObstacle[4]) == 0 and len(self.CapsicumPics) >= 3):
            self.numFruitList.append(int(i+1))
            self.fruits_list.append(PotentialObstacle[int(i)])
            self.EstimationAndMerge(self.CapsicumPics, self.CapsicumPics)

        
        
            # Estimate position
            # Use Target Pose Est
            # Set regeneration to true
            # Mark detected fruit as known fruit
    def EstimationAndMerge (self, pictureArray, poseArray):
        xarray = []
        yarray = []

        for i in range(len(pictureArray)):
            CurrentImage = PIL.image.fromarray(pictureArray[i])
            completed_img_dict = self.get_image_info(CurrentImage, poseArray[i])
            target_map = self.estimate_pose(self.camera_matrix, completed_img_dict)
            xarray.append(target_map['x'])
            yarray.append(target_map['y'])


        est_x = statistics.median(xarray)
        est_y = statistics.median(yarray)
            

                
        self.fruits_true_pos = np.append(self.fruits_true_pos,[est_x, est_y], axis=0)
                    

        print("Unknown Fruit Detected!!!")
        print(self.fruits_true_pos)
        print(self.fruits_list)


                

        
            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='friday_3_fruit.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()
    operate = Operate(args)
    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    
    operate.fruits_list, operate.fruits_true_pos, operate.aruco_true_pos = operate.read_true_map(args.map)
    operate.search_list = operate.read_search_list()
    operate.print_target_fruits_pos(operate.search_list, operate.fruits_list, operate.fruits_true_pos)

    # waypoints = [[0.8,0],[0.8,0.4]]
    reached_target = None
    target_num =0
    waypoints_num = 0
    not_scanning_around = False
    interlock_waypoints_num_3 = False

    operate.innitMarkers(operate.aruco_true_pos)

    fileK = "{}intrinsic.txt".format('./calibration/param/')
    # filedis = "{}distCoeffs.txt".format('./calibration/param/')
    operate.camera_matrix = np.loadtxt(fileK, delimiter=',')

    #Convert Search list to number
    for i in range(len(operate.search_list)):
        operate.numFruitList.append(operate.FruitsToNum_dict[operate.search_list[i]])
    print(operate.fruits_true_pos)

    while start:
        operate.update_keyboard()
        
        
        if operate.start:
            # operate.take_pic()
            if (waypoints_num%2 ==0 and waypoints_num!=0 and interlock_waypoints_num_3 == False) or reached_target == True or not_scanning_around == False:
                not_scanning_around = operate.turn_around(operate.get_robot_pose())
                interlock_waypoints_num_3 = True
                if not_scanning_around==True and waypoints_num==0 :
                    reached_target = None
                #print('dddd')
            else:
                if reached_target ==True or reached_target ==None or operate.RegenerateFlag:
                    operate.RegenerateFlag = False
                    print('fruit target')
                    print(operate.fruit_orderlist[target_num])
                    wayspoints = operate.optimum_waypoint_from_robotpos(operate.get_robot_pose(),operate.fruit_orderlist[target_num])
                    print('waypoints')
                    print(wayspoints)
                    reached_target =False
                
                nextwaypoint_bool = operate.control(wayspoints[waypoints_num],operate.get_robot_pose())

                if nextwaypoint_bool == True:
                    waypoints_num +=1
                    interlock_waypoints_num_3=False
                    if waypoints_num >= len(wayspoints):
                        waypoints_num=0
                        reached_target = True
                        not_scanning_around = False
                        target_num+=1

                        
                        if target_num>= len(operate.search_list):
                            operate.lv, operate.rv = operate.pibot.set_velocity([0, 0])
                            print('DONE moving to all fruits')
                            time.sleep(5)
                            pygame.quit()
                            sys.exit()

                        wait = True
                        print('press Enter to continue next fruit')
                        while wait:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    wait = False
                    print('wayspoints to')
                    print(wayspoints[waypoints_num])
                
            # time.sleep(7)
        
        #Turn on when testing outside
        operate.take_pic()
        operate.update_slam(measure.Drive(0, 0, 0.1))
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()





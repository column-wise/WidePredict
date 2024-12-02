"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import torch

from attrdict import AttrDict
from sgan.sgan.data.loader import data_loader
from sgan.sgan.models import TrajectoryGenerator
from sgan.sgan.losses import displacement_error, final_displacement_error
from sgan.sgan.utils import relative_to_abs, get_dset_path

###################################
import os
import numpy as np
os.environ['MPLCONFIGDIR'] = "/home/jetbot/Desktop/tensorrt_demos/matplot_dir"
#https://stackoverflow.com/questions/9827377/setting-matplotlib-mplconfigdir-consider-setting-mplconfigdir-to-a-writable-dir
import matplotlib
import matplotlib.pyplot as plt
from skimage import io

import glob
import time
from time import sleep
import argparse
from filterpy.kalman import KalmanFilter
import sort, numpy

import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

np.random.seed(0)


WINDOW_NAME_L = 'LEFT'
WINDOW_NAME_R = 'RIGHT'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


# initialize SGAN
checkpoint = torch.load('./sgan/scripts/models/sgan-models/zara2_12_model.pt')
generator = get_generator(checkpoint)
_args = AttrDict(checkpoint['args'])

frame_number = 1
x1=[]
y1=[]
x2=[]
y2=[]
aa, bb = [], []

N_frame = 8
vel = 12
acc = 2

speed = 100
slow = 20


# parameter for convert pixel coordinates to world coordinates
cameraMtx = np.load("./convert_coord/Original camera matrix.npy")
rvec = np.load("./convert_coord/RVec.npy")
tvec = np.load("./convert_coord/TVec.npy")
Cc = np.matrix([0, 0, 0]).T

# added classes including track_and_predict, convert_coord, sgan_predictioin
class Track_And_Predict:

    def __init__(self, line_min, line_max):
        self.curr_ped_seq = []
        self.ped_in_seq_len = np.array([], dtype=float).reshape(0,4)

        self.pred_list = [[0,0,0,0,0]]
        self.pred_flag = 0

        self.line_min = line_min
        self.line_max = line_max
        self.P = []

    def convert_coord(self, tracks_tmp):
       # tracks_tmp :
       # x_min, y_min, x_max, y_max, track_id

        x = (tracks_tmp[0] + tracks_tmp[2]) / 2
        y = tracks_tmp[3]

        # check npy file and get fx, fy, cx, cy values
        u = (x - 282)/476
        v = (y - 422)/371

        Pc = np.matrix([u, v, 1]).T
        Pw = rvec.T * (Pc - tvec)
        Cw = rvec.T * (Cc - tvec)

        k = Cw[2] / (Cw[2] - Pw[2])
        self.P = Cw + (Pw - Cw) * k

    def simple_predict(self, tracks_tmp, img):

        x_min=tracks_tmp[0]
        y_min=tracks_tmp[1]
        x_max=tracks_tmp[2]
        y_max=tracks_tmp[3]
        center_x = (x_min+x_max)/2
        center_y = y_max
        center_x = int(center_x - center_x % acc)
        center_y = int(center_y - center_y % acc)

        track_id = tracks_tmp[4]
        img = cv2.putText(img, str(track_id), (center_x, center_y),1, 1, (0,0,255), 2)

        append_flag = 1
        for i in range(0, len(self.pred_list)):
            if track_id == self.pred_list[i][2]:
                self.pred_list[i][3] = center_x
                self.pred_list[i][4] = center_y
                #print("\n pred list coord ", pred_list)
                x_var = center_x + (self.pred_list[i][3] - self.pred_list[i][0])*vel
                y_var = center_y + (self.pred_list[i][4] - self.pred_list[i][1])*vel
                img = cv2.arrowedLine(img, (center_x, center_y), (x_var, y_var), (0,255,0), 2, tipLength = 0.5)
                var_list = [x_var, 0, x_var, y_var]

                w_x = float(self.P[0])
                w_y = float(self.P[1])
                self.convert_coord(var_list)
                w_var_x = float(self.P[0])
                w_var_y = float(self.P[1])

                global speed

                if self.line_min > w_x and self.line_max >= w_y:
                    if speed > 0:
                           if 1/(7-w_y)*slow > 20:
                               speed -= 20
                           else:
                               speed -= 1/(7-w_y)*slow
                    else:
                        speed = 0
                elif self.line_min > w_x and self.line_max < w_y:
                    if speed > 0:
                           if 1/(7-w_y)*slow > 20:
                               speed -= 20
                           else:
                               speed -= 1/(7-w_y)*slow
                    else:
                        speed = 0

                elif self.line_min <= w_x and self.line_max < w_y:
                    if speed > 0:
                           if 1/(7-w_y)*slow > 20:
                               speed -= 20
                           else:
                               speed -= 1/(7-w_y)*slow
                    else:
                        speed = 0

                global frame_number

                # remove object ended prediction
                tmp_remove = self.pred_list.pop(i)
                append_flag = 0
                break

        if append_flag == 1:
            self.pred_list.append([center_x, center_y, int(track_id),0,0])
            append_flag = 0



    def track_and_predict(self, boxes, confs, clss, img, tracker_):
        dets_to_sort = np.empty((0,5))
        for i, boxes_tmp in enumerate(boxes):
            if clss[i] == 0:
                boxes_tmp = np.append(boxes_tmp[:],confs[i])
                dets_to_sort = np.vstack((dets_to_sort, boxes_tmp))
        tracks = tracker_.update(dets_to_sort)
        # tracks contains (x_min, y_min, x_max, y_max, id)

        tracks_tmp = []

        self.curr_ped_seq = []
        global frame_number

        for tracks_tmp in tracks:
            if frame_number > 0 : # change initial frame_number to -1, to skip not having correct tracking information
               self.convert_coord(tracks_tmp)
               self.curr_ped_seq.append([float(frame_number), float(tracks_tmp[4]), float(self.P[0]),float(self.P[1])])

               if self.P[0] > self.line_min and self.P[0] < self.line_max:
                   global speed
                   if speed > 0:
                       if 1/(7-self.P[1])*slow > 20:
                           speed -= 20
                       else:
                           speed -= 1/(7-self.P[1])*slow
                   else:
                       speed = 0

            self.simple_predict(tracks_tmp, img)

        self.pred_flag = self.pred_flag + 1
        if self.pred_flag >= N_frame:

            # initialize pred_list_R
            self.pred_list=[[0,0,0,0,0]]
            self.pred_flag = 0

        self.sgan_prediction()
        print("\nspeed =", speed)
        return img

    def sgan_prediction(self):
        global generator, _args, frame_number
        if len(self.curr_ped_seq) is 0:
           self.ped_in_seq_len = np.array([], dtype=float).reshape(0,4)
        else:
           self.ped_in_seq_len = np.vstack([self.ped_in_seq_len, np.asarray(self.curr_ped_seq)])
        frames = np.unique(self.ped_in_seq_len[:, 0]).tolist()
        if len(frames) == 8:
            peds_in_curr_seq = np.unique(self.ped_in_seq_len[:, 1])
            considered_ped = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                self.curr_ped_seq = self.ped_in_seq_len[self.ped_in_seq_len[:, 1] == ped_id, :]
                if self.curr_ped_seq.shape[0] == 8:
                    considered_ped.append(ped_id)
            if len(considered_ped) > 0:
                _, loader = data_loader(_args, self.ped_in_seq_len)
                for batch in loader:
                    batch = [tensor.cuda() for tensor in batch]
                    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                     non_linear_ped, loss_mask, seq_start_end) = batch
                    start = time.time()
                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )
                    pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel, obs_traj[-1]
                    )

                    tmp = len(considered_ped)
                    for i in range(0, tmp):
                        gt = pred_traj_gt[:, i, :].data
                        input_a = obs_traj[:, i, :].data
                        out_a = pred_traj_fake[:, i, :].data
                        aa = np.concatenate((input_a.cuda().data.cpu().numpy(), gt.cuda().data.cpu().numpy()), axis=0)
                        bb = np.concatenate((input_a.cuda().data.cpu().numpy(), out_a.cuda().data.cpu().numpy()), axis=0)
                        global x1, y1

                        plt.ylim([15,-20])
                        plt.xlim([-12,20])

                        x1 = bb[:, 0]
                        y1 = bb[:, 1]

                        plt.plot(aa[:,0], aa[:,1], '*')
                        plt.plot(bb[:,0], bb[:,1], '-')

                    plt.vlines(self.line_min, 7,-7, colors='blue', linestyle='solid')
                    plt.vlines(self.line_max, 7,-7, colors='blue', linestyle='solid')


                plt.show()
                self.ped_in_seq_len = np.array([], dtype=float).reshape(0, 4)
                frame_number = -1
            else:
                self.ped_in_seq_len = np.array([], dtype=float).reshape(0, 4)
                frame_number = -1
M = []
dim = []

def ORB_variable(cam_L, cam_R):

    capture_L = cam_L
    capture_R = cam_R

    while True:

        left = capture_L.read()
        right = capture_R.read()

        orb = cv2.ORB_create()

        kp_left, des_left = orb.detectAndCompute(left, None)
        kp_right, des_right = orb.detectAndCompute(right, None)

        keypoints_drawn_left = cv2.drawKeypoints(left, kp_left, None, color=(0, 0, 255))
        keypoints_drawn_right = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_left,des_right)

        matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, matches, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        limit = 10
        best = sorted(matches, key = lambda x:x.distance)[:limit]

        best_matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, best, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        left_pts = []
        right_pts = []
        for m in best:
            l = kp_left[m.queryIdx].pt
            r = kp_right[m.trainIdx].pt
            left_pts.append(l)
            right_pts.append(r)

        global M, dim
        M, _ = cv2.findHomography(np.float32(right_pts), np.float32(left_pts))

        dim_x = left.shape[1] + right.shape[1]
        dim_y = max(left.shape[0], right.shape[0])
        dim = (dim_x, dim_y)

        while True:

            left = capture_L.read()
            right = capture_R.read()

            cv2.imshow("left", left)
            cv2.imshow("right",right)

            warped = cv2.warpPerspective(right, M, dim)

            # finally we cat put the two images together.
            comb = warped.copy()
            # combine the two images
            comb[0:left.shape[0],0:left.shape[1]] = left
            # crop
            r_crop = 850
            comb = comb[:, :r_crop]

            cv2.imshow('comb',comb)
            if cv2.waitKey(1) == 27:
                cv2.destroyWindow('left')
                cv2.destroyWindow('right')
                cv2.destroyWindow('comb')
                return 0
            elif cv2.waitKey(1) == ord('r'):
                break

def ORB_stitch(img_L, img_R):

    left = img_L
    right = img_R

    global M, dim
    warped = cv2.warpPerspective(right, M, dim)
    comb = warped.copy()
    comb[0:left.shape[0],0:left.shape[1]] = left
    r_crop = 900
    comb = comb[:, :r_crop]

    return comb

def loop_and_detect(cam_L, cam_R, trt_yolo, tracker, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    trk_prd = Track_And_Predict(1.5, 7)
    print("\n press ESC or press 'r' ")
    print("\n press 'r' long enough ")
    _ = ORB_variable(cam_L, cam_R)

    print("prediction start in 5 seconds\n")
    sleep(1)
    print("prediction start in 4 seconds\n")
    sleep(1)
    print("prediction start in 3 seconds\n")
    sleep(1)
    print("prediction start in 2 seconds\n")
    sleep(1)
    print("prediction start in 1 seconds\n")
    sleep(1)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME_L, 0) < 0:
            break
        if cv2.getWindowProperty(WINDOW_NAME_R, 0) < 0:
            break

        img_L = cam_L.read()
        img_R = cam_R.read()
        global frame_number
        frame_number += 1

        if img_L is None:
            break
        if img_R is None:
            break
        start_e = time.time()

        img = ORB_stitch(img_L, img_R)

        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = trk_prd.track_and_predict(boxes, confs, clss, img, tracker)

        img = show_fps(img, fps)

        img = cv2.line(img, (250,480), (320,400), (255,0,0), 5)
        img = cv2.line(img, (500,480), (440,400), (255,0,0), 5)

        cv2.imshow(WINDOW_NAME_L, img_L)
        cv2.imshow(WINDOW_NAME_R, img_R)
        cv2.imshow('stitched image', img)


        toc = time.time()
        curr_fps = 1.0 / (toc - tic)

        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display("FULL SCREEN", full_scrn)

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam_L = Camera(args)
    if not cam_L.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # to set device/video1
    args.onboard = 0
    cam_R = Camera(args)
    if not cam_R.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    tracker = sort.Sort(max_age=0.5, min_hits=1, iou_threshold=0.4)

    open_window(
        WINDOW_NAME_R, 'Camera R TensorRT YOLO Demo',
        cam_L.img_width, cam_L.img_height)
    open_window(
        WINDOW_NAME_L, 'Camera L TensorRT YOLO Demo',
        cam_L.img_width, cam_L.img_height)

    loop_and_detect(cam_L, cam_R, trt_yolo, tracker, conf_th=0.3, vis=vis)

    cam_L.release()
    cam_R.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

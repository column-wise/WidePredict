import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

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

xdata, ydata = [], []
x1=[]
y1=[]
aa, bb = [], []
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'ro')

def gen_dot():
    for i in range(0,len(x1)):
        newdot = [x1[i], y1[i]]
        yield newdot

def update_dot(newd):
    xdata.append(newd[0])
    ydata.append(newd[1])
    ln.set_data(xdata, ydata)
    return ln


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # values for convert pixel coordinates to world coordinates
    cameraMtx = np.load("test/Original camera matrix.npy")
    rvec = np.load("test/RVec.npy")
    tvec = np.load("test/TVec.npy")
    Cc = np.matrix([0, 0, 0]).T

    # initialize SGAN
    checkpoint = torch.load('models/sgan-models/01/zara2_12_model.pt')
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    ped_in_seq_len = np.array([], dtype=float).reshape(0,4)

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        # allowed_classes = ["bicycle"]

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # call the tracker
        tracker.predict()
        tracker.update(detections)

        curr_ped_seq = []

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # bbox = min x, min y, max x, max y
            bbox = track.to_tlbr()

            # pixel coordinate (u, v) to convert
            u = (bbox[0] + bbox[2]) / 2
            v = bbox[3]

            Pc = np.matrix([u, v, 1]).T
            Pw = rvec.T * (Pc - tvec)
            Cw = rvec.T * (Cc - tvec)

            k = Cw[2] / (Cw[2] - Pw[2])
            P = Cw + (Pw - Cw) * k

            curr_ped_seq.append([float(frame_num), float(track.track_id), float(P[0]),float(P[1])])

            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id) + ' x: '+str(float(P[0])) + ' y: ' + str(float(P[1])),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if len(curr_ped_seq) is 0:
            ped_in_seq_len = np.array([], dtype=float).reshape(0,4)
        else:
            ped_in_seq_len = np.vstack([ped_in_seq_len, np.asarray(curr_ped_seq)])

        frames = np.unique(ped_in_seq_len[:, 0]).tolist()
        if len(frames) == 8:
            peds_in_curr_seq = np.unique(ped_in_seq_len[:, 1])
            considered_ped = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = ped_in_seq_len[ped_in_seq_len[:, 1] ==
                                             ped_id, :]

                if curr_ped_seq.shape[0] == 8:
                    considered_ped.append(ped_id)

            print('considered_ped:', len(considered_ped))

            _, loader = data_loader(_args, ped_in_seq_len)

            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                 non_linear_ped, loss_mask, seq_start_end) = batch

                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                gt = pred_traj_gt[:, 0, :].data

                input_a = obs_traj[:, 0, :].data
                out_a = pred_traj_fake[:, 0, :].data
                aa = np.concatenate((input_a.cuda().data.cpu().numpy(), gt.cuda().data.cpu().numpy()), axis=0)
                bb = np.concatenate((input_a.cuda().data.cpu().numpy(), out_a.cuda().data.cpu().numpy()), axis=0)
                global x1, y1

                ax.set_xlim(np.min(aa[:, 0] - 1), np.max(aa[:, 0]) + 1)
                ax.set_ylim(np.min(aa[:, 1] - 1), np.max(aa[:, 1]) + 1)

                x1 = bb[:, 0]
                y1 = bb[:, 1]
                l = ax.plot(aa[:, 0], aa[:, 1], '.')
                ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=50)
                plt.show()
                plt.close()

            for track in tracker.tracks:
                considered_ped = np.array(considered_ped)
                pred_coord = pred_traj_fake[:, np.where(considered_ped == track.track_id), :]

                if pred_coord.size()[2] != 0:
                    pred_coord = pred_coord.reshape(12, 2)
                    pred_coord = pred_coord.cpu().detach().numpy()

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]

                    for i in range(12):

                        # P = (x, y, 0.0)
                        P = np.append(pred_coord[i], np.array([0.0]))

                        Pxyz1 = np.insert(P, 3, 1).reshape(4, 1)

                        xy = cameraMtx * (np.mat(np.append(rvec, tvec, axis=1)) * np.mat(Pxyz1))
                        xy = xy / xy[2]

                        u = (xy[0] - cameraMtx[0][2]) / cameraMtx[0][0]
                        v = (xy[1] - cameraMtx[1][2]) / cameraMtx[1][1]

                        cv2.circle(frame, (u, v), 3, color, 5)

            ped_in_seq_len = np.array([], dtype=float).reshape(0, 4)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
            cv2.waitKey(0)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

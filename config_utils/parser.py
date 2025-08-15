import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Light YOLO Training")

    # default.yaml / parameter to load the model
    parser.add_argument("--config", type=bool, default=True,
                        help="method to load the model, i.e. true or false to load the model by this way or by default.yaml")
    # Task and Mode
    parser.add_argument('--task', type=str, default='detect',
                        help='YOLO task, i.e. detect, segment, classify, pose, obb')
    parser.add_argument('--mode', type=str, default='train',
                        help='YOLO mode, i.e. train, val, predict, export, track, benchmark')

    # Train settings
    parser.add_argument('--model', type=str, default="", help='path to model file, i.e. yolo11n.pt, yolov8n.yaml')
    parser.add_argument('--data_path', type=str, default='./datasets', help='path to datasets file')
    parser.add_argument('--data', type=str, default='data_big.yaml', help='path to data file, i.e. coco8.yaml')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--time', type=float, default=None,
                        help='number of hours to train for, overrides epochs if supplied')
    parser.add_argument('--patience', type=int, default=5,
                        help='epochs to wait for no observable improvement for early stopping of training')
    parser.add_argument('--batch', type=int, default=32, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640],
                        help='input images size as int for train and val modes, or list[h,w] for predict and export modes')
    parser.add_argument('--save', action='store_true', default=True, help='save train checkpoints and predict results')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', default=False, action='store_true',
                        help='True/ram, disk or False. Use cache for data loading')
    parser.add_argument('--device', type=str, default=[0, 1],
                        help='device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of worker threads for data loading (per RANK if DDP)')
    parser.add_argument('--project', type=str, default=None, help='project name')
    parser.add_argument('--name', type=str, default=None,
                        help='experiment name, results saved to project/name directory')
    parser.add_argument('--exist_ok', default=False, action='store_true',
                        help='whether to overwrite existing experiment')
    parser.add_argument('--pretrained', type=str, default=True,
                        help='whether to use a pretrained model (bool) or a model to load weights from (str)')
    parser.add_argument('--optimizer', type=str, default='auto',
                        help='optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]')
    parser.add_argument('--verbose', default=True, action='store_true', help='whether to print verbose output')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--deterministic', default=True, action='store_true',
                        help='whether to enable deterministic mode')
    parser.add_argument('--single_cls', default=False, action='store_true',
                        help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', default=False,
                        help='rectangular training if mode=train or rectangular validation if mode=val')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10,
                        help='disable mosaic augmentation for final epochs (0 to disable)')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile ONNX and TensorRT speeds during training for loggers')
    parser.add_argument('--freeze', type=str, default=None,
                        help='freeze first n layers, or freeze list of layer indices during training')
    parser.add_argument('--multi_scale', action='store_true', default=False,
                        help='Whether to use multiscale during training')

    # Segmentation specific
    parser.add_argument('--overlap_mask', action='store_true', default=True,
                        help='merge object masks into a single image mask during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')

    # Classification specific
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

    # Val/Test settings
    parser.add_argument('--val', action='store_true', default=True, help='validate/test during training')
    parser.add_argument('--split', type=str, default='val',
                        help='dataset split to use for validation, i.e. val, test or train')
    parser.add_argument('--save_json', action='store_true', default=True, help='save results to JSON file')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, default=None,
                        help='object confidence threshold for detection (default 0.25 predict, 0.001 val)')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--half', action='store_true', default=True, help='use half precision (FP16)')
    parser.add_argument('--dnn', action='store_true', default=False, help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', default=True, action='store_true', help='save plots and images during train/val')

    # Predict settings
    parser.add_argument('--source', type=str, default=None, help='source directory for images or videos')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', action='store_true',
                        help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--visualize', action='store_true', help='visualize model features')
    parser.add_argument('--augment', action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='filter results by class, i.e. classes=0, or classes=[0,2,3]')
    parser.add_argument('--retina_masks', action='store_true', help='use high-resolution segmentation masks')
    parser.add_argument('--embed', type=int, nargs='+', default=None,
                        help='return feature vectors/embeddings from given layers')

    # Visualize settings
    parser.add_argument('--show', action='store_true', help='show predicted images and videos if environment allows')
    parser.add_argument('--save_frames', action='store_true', help='save predicted individual video frames')
    parser.add_argument('--save_txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels', action='store_true', help='show prediction labels, i.e. person')
    parser.add_argument('--show_conf', action='store_true', help='show prediction confidence, i.e. 0.99')
    parser.add_argument('--show_boxes', action='store_true', help='show prediction boxes')
    parser.add_argument('--line_width', type=int, default=None,
                        help='line width of the bounding boxes. Scaled to image size if None.')

    # Export settings
    parser.add_argument('--format', type=str, default='torchscript',
                        help='format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats')
    parser.add_argument('--keras', action='store_true', help='use Keras')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model using onnxslim')
    parser.add_argument('--opset', type=int, default=None, help='ONNX: opset version')
    parser.add_argument('--workspace', type=float, default=None,
                        help='TensorRT: workspace size (GiB), None will let TensorRT auto-allocate memory')
    parser.add_argument('--nms', action='store_true', help='CoreML: add NMS')

    # Hyperparameters
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate (i.e. SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay 5e-4')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='warmup epochs (fractions ok)')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='warmup initial bias lr')
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain (scale with pixels)')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--pose', type=float, default=12.0, help='pose loss gain')
    parser.add_argument('--kobj', type=float, default=1.0, help='keypoint obj loss gain')
    parser.add_argument('--nbs', type=int, default=64, help='nominal batch size')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='image perspective (+/- fraction), range 0-0.001')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--bgr', type=float, default=0.0, help='image channel BGR (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='segment copy-paste (probability)')
    parser.add_argument('--copy_paste_mode', type=str, default='flip',
                        help='the method to do copy_paste augmentation (flip, mixup)')
    parser.add_argument('--auto_augment', type=str, default='randaugment',
                        help='auto augmentation policy for classification (randaugment, autoaugment, augmix)')
    parser.add_argument('--erasing', type=float, default=0.4,
                        help='probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0.')
    parser.add_argument('--crop_fraction', type=float, default=1.0,
                        help='image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0.')

    # Custom config.yaml
    parser.add_argument('--cfg', type=str, default=None, help='for overriding defaults.yaml')

    # Tracker settings
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                        help='tracker type, choices=[botsort.yaml, bytetrack.yaml')

    return parser.parse_args()

#!/usr/bin/env python3
"""Vehicle detection using YOLOv8."""
import sys
import cv2
import numpy as np
import traceback
from os.path import splitext, basename, isdir
from os import makedirs
from glob import glob
from ultralytics import YOLO
from src.label import Label, lwrite
from src.utils import crop_region


def image_files_from_folder(folder, extensions=('jpg', 'jpeg', 'png')):
    files = []
    for ext in extensions:
        files += glob('%s/*.%s' % (folder, ext))
        files += glob('%s/*.%s' % (folder, ext.upper()))
    return sorted(files)


if __name__ == '__main__':
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        vehicle_threshold = 0.5

        # COCO dataset classes: car=2, bus=5, truck=7, motorcycle=3
        VEHICLE_CLASSES = [2, 5, 7]
        VEHICLE_NAMES = {2: 'car', 5: 'bus', 7: 'truck'}

        model = YOLO('yolov8n.pt')

        imgs_paths = image_files_from_folder(input_dir)

        if not isdir(output_dir):
            makedirs(output_dir)

        print('Searching for vehicles using YOLOv8...')

        for img_path in imgs_paths:
            print('\tScanning %s' % img_path)
            bname = basename(splitext(img_path)[0])

            results = model(img_path, conf=vehicle_threshold, classes=VEHICLE_CLASSES, verbose=False)

            Iorig = cv2.imread(img_path)
            WH = np.array(Iorig.shape[1::-1], dtype=float)
            Lcars = []

            if len(results) and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    tl = np.array([x1 / WH[0], y1 / WH[1]])
                    br = np.array([x2 / WH[0], y2 / WH[1]])
                    label = Label(0, tl, br)
                    Icar = crop_region(Iorig, label)
                    Lcars.append(label)
                    cv2.imwrite('%s/%s_%dcar.png' % (output_dir, bname, i), Icar)

                lwrite('%s/%s_cars.txt' % (output_dir, bname), Lcars)
                print('\t\t%d vehicles found' % len(Lcars))
            else:
                print('\t\tNo vehicles found')

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)

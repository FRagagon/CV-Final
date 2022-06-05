import glob
import json
import math
import operator
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content



def get_map(MINOVERLAP, draw_plot, path = './map_out'):
    GT_PATH             = os.path.join(path, 'ground-truth')
    DR_PATH             = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH  = os.path.join(path, 'results')

    show_animation = False
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
        
    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class     = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id     = txt_file.split(".txt", 1)[0]
        file_id     = os.path.basename(os.path.normpath(file_id))
        temp_path   = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list      = file_lines_to_list(txt_file)
        bounding_boxes  = []
        is_difficult    = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split  = line.split()
                    _difficult  = line_split[-1]
                    bottom      = line_split[-2]
                    right       = line_split[-3]
                    top         = line_split[-4]
                    left        = line_split[-5]
                    class_name  = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name  = class_name[:-1]
                    is_difficult = True
                else:
                    line_split  = line.split()
                    bottom      = line_split[-1]
                    right       = line_split[-2]
                    top         = line_split[-3]
                    left        = line_split[-4]
                    class_name  = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes  = list(gt_counter_per_class.keys())
    gt_classes  = sorted(gt_classes)
    n_classes   = len(gt_classes)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split      = line.split()
                    bottom          = line_split[-1]
                    right           = line_split[-2]
                    top             = line_split[-3]
                    left            = line_split[-4]
                    confidence      = line_split[-5]
                    tmp_class_name  = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name  = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    lamr_dictionary = {}
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd          = len(dr_data)
            tp          = [0] * nd
            fp          = [0] * nd
            score       = [0] * nd
            score05_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id     = detection["file_id"]
                score[idx]  = float(detection["confidence"])
                if score[idx] > 0.5:
                    score05_idx = idx

                gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data   = json.load(open(gt_file))
                ovmax       = -1
                gt_match    = -1
                bb          = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt    = [ float(x) for x in obj["bbox"].split() ]
                        bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw      = bi[2] - bi[0] + 1
                        ih      = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                        else:
                            fp[idx] = 1
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

               
            

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP  += ap

        results_file.write("\n# mAP of all classes\n")
        mAP     = sum_AP / n_classes
        text    = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    
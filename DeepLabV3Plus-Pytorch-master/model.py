import cv2
import os
from PIL import Image

# turn video into pictures

def v_to_p(video_path, pic_path):
    cap =  cv2.VideoCapture("video_path")

    frame_count = 0
    while True:
        frame_count += 1
        flag, frame = cap.read()
        if not flag or frame_count >= 3600:
            break
        cv2.imwrite(r"pic_path\{}.png".format(frame_count), frame)
    cap.release()

def p_to_v(video_path, pic_path):
    image_names = os.listdir(pic_path)
    image_names.sort(key=lambda x:int(x[:-4]))
    fourcc = cv2.VideoWriter_fourcc("M","P","4","V")
    fps = 30
    image = Image.open(os.path.join(pic_path,image_names[0]))
    video_writer = cv2.VideoWriter(video_path,fourcc,fps,image.size)
    for i in image_names:
        im = cv2.imread(os.path.join(pic_path,i))
        video_writer.write(im)
    video_writer.release()

p_to_v("video_result.mp4", "test_results")
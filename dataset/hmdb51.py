import os
import cv2
import shutil
import pandas as pd
from natsort import natsorted


def split_hmdb51_data(video_root_path, annot_dir, split_dir):
    count_videos = 0
    for video_dir in os.listdir(video_root_path):
        video_path = os.path.join(video_root_path, video_dir)
        for annot_file in os.listdir(annot_dir):
            annot_path = os.path.join(annot_dir, annot_file)
            action_name = video_path.split('/')[-1]
            action_name2 = annot_path.split('/')[-1].split('.')[0][:-12]  # _test_split1
            if action_name == action_name2:
                # print(action_name, " <==> ", action_name2)
                with open(annot_path, 'r') as f:
                    for line in f:
                        abs_video_path = os.path.join(video_path, line.strip().split(' ')[0])  #

                        labels = int(line.strip().split(' ')[-1])
                        if labels != 1 and labels != 0: 
                            count_videos += 1
                            split_folder = os.path.join(split_dir, action_name)
                            os.makedirs(split_folder, exist_ok=True)

                            shutil.copy(abs_video_path, split_folder)
                            print('Copying videos from [{}] action class...'.format(action_name))
        print('number of train videos for {} are : {}'.format(video_path.split('/')[-1], str(count_videos)))
        count_videos = 0 


def generate_hmdb51_video_list(root_dir, filename):
    video_list = []
    id_no = 1
    for cat in os.listdir(root_dir):
        video_path = os.path.join(root_dir, cat)
        for video in os.listdir(video_path):
            video_list.append([video, cat, int(id_no)])
        id_no += 1
    video_file = pd.DataFrame(video_list, columns=['video_file', 'label', 'target'], index=None)
    video_file.to_csv(filename)
    print('Video files are written to csv file')


def detect_anomaly_clips(video_dir, dest_dir):
    i = 0
    j = 0
    k = 0
    for video in os.listdir(video_dir):
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        assert (cap.isOpened()), "\nInvalid video path input >> {}".format(vid_path)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if height < 112 or width < 112:
            print('Moving: {}, due to smaller dimension====================='.format(os.path.join(video_dir, video)))
            shutil.move(os.path.join(video_dir, video), dest_dir)
            k += 1
        if num_frames < 16:
            print('Moving: {}, due to less number of frames'.format(os.path.join(video_dir, video)))
            shutil.move(os.path.join(video_dir, video), dest_dir)
            i += 1
        else:
            j += 1
            continue
    print('{} videos are moved.'.format(str(i)))
    print('{} videos are above the threshold.'.format(str(j)))

if __name__ == '__main__':
    split_path = "/media/ican/XxX/Datasets/HMDB51/splitted/test10"
    split_csv_file = "test10split01.csv"
    generate_hmdb51_video_list(split_path, split_csv_file)

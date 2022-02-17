import os
from re import I
import cv2
import shutil
import pandas as pd
from natsort import natsorted
from natsort.ns_enum import ns


def split_ucf101_data(video_dir, annot_file, split_dir):
    with open(annot_file, 'r') as f:
        for line in f:
            class_path = line.strip().split(' ')[0]  
            class_name = class_path.split('/')[0] 
            video = class_path.split('/')[-1] 
           
            split_folder = os.path.join(split_dir, class_name)
            os.makedirs(split_folder, exist_ok=True)  
            abs_video_path = os.path.join(video_dir, class_path)
            
            try:
                shutil.copy(abs_video_path, split_folder)
                print('Copying videos from [{}] action class...'.format(class_name))
            
            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            
            # If destination is a directory.
            except IsADirectoryError:
                print("Destination is a directory.")
            
            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")
            
            # For other errors
            except:
                print("Error occurred while copying file.")


def generate_ucf101_video_list(root_dir, filename):
    video_list = []
    id_no = 0
    for cat in os.listdir(root_dir):
        video_path = os.path.join(root_dir, cat)
        for video in os.listdir(video_path):
            video_list.append([video.split('.')[0], video, cat, int(id_no)])
        id_no += 1
    video_file = pd.DataFrame(video_list, columns=['video_dir', 'video_file', 'label', 'target'], index=None)
    video_file.to_csv(filename)
    print('Video files are written to csv file')


def detect_anomaly_clips(video_dir, dest_dir):
    i = 0
    j = 0
    k = 0
    for video in os.listdir(video_dir):
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        assert (cap.isOpened()), "\nInvalid video path input >> {}".format(video_dir)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if height < 224 or width < 224:
            print('Moving: {}, due to smaller dimension'.format(os.path.join(video_dir, video)))
            shutil.move(os.path.join(video_dir, video), dest_dir)
            k += 1
        if num_frames < 36:
            print('Moving: {}, due to less number of frames'.format(os.path.join(video_dir, video)))
            shutil.move(os.path.join(video_dir, video), dest_dir)
            i += 1
        else:
            j += 1
            continue
    print('{} videos are moved.'.format(str(i)))
    print('{} videos are above the threshold.'.format(str(j)))


def move_files(root_dir, dest_dir):
    k = 0
    for i, cat in enumerate(os.listdir(root_dir)):
        video_path = os.path.join(root_dir, cat)
        for j, video in enumerate(os.listdir(video_path)):
            shutil.move(os.path.join(video_path, video), dest_dir)
            k += 1
        print("Copying: {} videos...".format(cat))

    print('All video files are: {}, copied to path: {}.'.format(k, dest_dir.upper()))


def generate_video_dir(video_list, root_dir=None, dest_dir=None, filename='trainingfile.csv'):
    list_of_video_files = pd.read_csv(video_list)
    video_files = list_of_video_files['video_file']
    labels = list_of_video_files['label']
    targets = list_of_video_files['target']

    list_of_videos_with_frame_loc = []
    for i, video in enumerate(video_files):
        abs_video_path = os.path.join(root_dir, video)
        dest_path = os.path.join(dest_dir, video.split('.')[0])
        os.makedirs(dest_path, exist_ok=True)
        
        list_of_videos_with_frame_loc.append([video.split('.')[0], video, labels[i], targets[i]])
        # open video file
        cap = cv2.VideoCapture(abs_video_path)
        assert (cap.isOpened())
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(num_frames):
            # read frame
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(dest_path + "/" + "frame_{}.jpg".format(str(frame_idx+1)), frame)
        cap.release()

    video_file = pd.DataFrame(list_of_videos_with_frame_loc, columns=['video_dir','video_file', 'label', 'target'], index=None)
    video_file.to_csv(filename)
    print('Video files are written to csv file')


def generate_video_frames(path2video_list, dest_dir):

    for video in os.listdir(path2video_list):
        print("Extracting frames from: {}".format(video))
        abs_video_path = os.path.join(path2video_list, video)
        dest_path = os.path.join(dest_dir, video.split('.')[0])
        os.makedirs(dest_path, exist_ok=True)
        # open video file
        cap = cv2.VideoCapture(abs_video_path)
        assert (cap.isOpened())
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(num_frames):
            # read frame
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(dest_path + "/" + "{}_frame_{}.jpg".format(video.split('.')[0], str(frame_idx)), frame)
        cap.release()

    print("Total video samples: {}".format(len(os.listdir(path2video_list))))
    print('Frame extraction completed!')


def count_num_of_frames(path2videoframes):
    count_videos = 0
        
    i = 0
    for video_dir in os.listdir(path2videoframes):
        frame_path = os.path.join(path2videoframes, video_dir)
        frame_length = len(os.listdir(frame_path))
        if frame_length < 48:
            print("Video: {} ==> {}".format(video_dir, frame_length))
            i += 1
        count_videos += 1

    print("Total videos containing less than 48: {}, from total frames of: {}".format(str(i), count_videos))


if __name__ == '__main__':
    video_frame_path = "./dataset/ucf101_for_semi_supervised/ucf101frames"
    count_num_of_frames(video_frame_path)

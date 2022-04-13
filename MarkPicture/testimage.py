# import datetime
#
# import imageio
# import pylab
# import skimage
# import numpy as np
#
# filename = 'E:\衡阳到长沙\衡阳-岳阳.mp4'
# vid = imageio.get_reader(filename, 'ffmpeg')
# frame_list = []
# for num, im in enumerate(vid):
#     time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
#     print("Start : %s" % time_now)
#     print(num)
#     time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
#     print("END : %s" % time_now)
import datetime

import ffmpeg
import numpy
import cv2
import sys
import random


def read_frame_as_jpeg(in_file, frame_num):
    """
    指定帧数读取任意帧
    """
    out, err = (
        ffmpeg.input(in_file)
              .filter('select', 'gte(n,{})'.format(frame_num))
              .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
              .run(capture_stdout=True)
    )
    return out


def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


if __name__ == '__main__':
    file_path = 'E:/衡阳到长沙/衡阳-岳阳.mp4'
    # video_info = get_video_info(file_path)
    # total_frames = int(video_info['nb_frames'])
    # print('总帧数：' + str(total_frames))
    # random_frame = random.randint(1, total_frames)
    # print('随机帧：' + str(random_frame))
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("Start : %s" % time_now)
    out = read_frame_as_jpeg(file_path, 50)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("MID : %s" % time_now)
    image_array = numpy.asarray(bytearray(out), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("END : %s" % time_now)
    cv2.imshow('frame', image)
    cv2.waitKey()
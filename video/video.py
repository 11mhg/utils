import numpy as np
import os, time, random
import cv2


class Video():

    def __init__(self,framerate,filenames):
        self.framerate = framerate
        self.msperframe = 500.//self.framerate
        self.filenames = filenames
        self.frames = []
        for filename in self.filenames:
            self.frames.append(cv2.imread(filename,cv2.IMREAD_COLOR))
        self.length = len(self.frames)//self.framerate


    def play(self):
        while True:
            for frame_ind in range(len(self.frames)):
                frame = self.frames[frame_ind]
                cv2.imshow('Frame',frame)
                if cv2.waitKey(int(self.msperframe)) and 0XFF == ord('Q'):
                    break
            if cv2.waitKey(5000) and 0XFF == ord('K'):
                break
        cv2.destroyAllWindows()

from tqdm import tqdm
import numpy as np
import re
import os, time, random
import cv2

def sort_nicely( l ): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l,key=alphanum_key)


class Video():

    def __init__(self,framerate,filenames=None,frames = None):
        self.framerate = framerate
        self.msperframe = 1000.//self.framerate
        if filenames is not None:
            self.filenames = [i for i in filenames if '.jpg' in i]
            self.filenames = sort_nicely(self.filenames)
            self.frames = []
            for filename in tqdm(self.filenames):
                try:
                    self.frames.append(cv2.imread(filename,cv2.IMREAD_COLOR))
                except:
                    print("file could not be loaded: {}".format(filename))   
            self.frames = np.array(self.frames)
            self.length = len(self.frames)//self.framerate
        if frames is not None:
            self.frames = np.array(frames)
            self.length = len(self.frames)//self.framerate

    def start_write(self,out_path):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        self.out = cv2.VideoWriter(out_path,fourcc,float(self.framerate),(416,416))
        return 

    def write_frame(self,images):
        if len(images.shape)> 3:
            for ind in range(images.shape[0]):
                image = cv2.cvtColor(np.array(images[ind,:,:,:],dtype=np.uint8),cv2.COLOR_RGB2BGR)
                self.out.write(image)
        else:
            images =np.array(images,dtype=np.uint8)
            images = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
            self.out.write(images)
        return

    def close_write(self):
        self.out.release()
        return


    def play(self):
        for frame_ind in range(len(self.frames)):
            frame = self.frames[frame_ind]
            cv2.imshow('Frame',frame)
            if cv2.waitKey(int(self.msperframe)) and 0XFF == ord('Q'):
                break
        cv2.destroyAllWindows()

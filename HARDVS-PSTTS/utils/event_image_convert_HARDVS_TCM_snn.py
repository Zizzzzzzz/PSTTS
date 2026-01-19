# Read .aedat4 file from DAVIS346 at the temporal window (t1-t2), generating frames.

import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import json
import tqdm as tq
from concurrent.futures import ThreadPoolExecutor
import random
from numpy.lib import recfunctions
import shutil
import torch
import torch as th
import snntorch as snn
import torch.nn as nn
def is_int_tensor(tensor: th.Tensor) -> bool:
    return not th.is_floating_point(tensor) and not th.is_complex(tensor)
def merge_channel_and_bins(representation: th.Tensor):
    height = 260
    width = 346
    assert representation.dim() == 4
    return th.reshape(representation, (-1, height, width))

def construct(x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
    device = x.device
    assert y.device == pol.device == time.device == device
    assert is_int_tensor(x)
    assert is_int_tensor(y)
    assert is_int_tensor(pol)
    assert is_int_tensor(time)

    # dtype = th.uint8 if self.fastmode else th.int16
    dtype = th.int16
    channels = 2
    bins = 10
    height = 260
    width = 346
    count_cutoff = 10
    representation = th.zeros((channels, bins, height, width),
                                dtype=dtype, device=device, requires_grad=False)

    if x.numel() == 0:
        assert y.numel() == 0
        assert pol.numel() == 0
        assert time.numel() == 0
        return merge_channel_and_bins(representation.to(th.uint8))
    assert x.numel() == y.numel() == pol.numel() == time.numel()

    assert pol.min() >= 0
    assert pol.max() <= 1

    bn, ch, ht, wd = bins, channels, height, width

    # NOTE: assume sorted time
    t0_int = time[0]
    t1_int = time[-1]
    assert t1_int >= t0_int
    t_norm = time - t0_int
    t_norm = t_norm / max((t1_int - t0_int), 1)
    t_norm = t_norm * bn
    t_idx = t_norm.floor()
    t_idx = th.clamp(t_idx, max=bn - 1)

    indices = x.long() + \
                wd * y.long() + \
                ht * wd * t_idx.long() + \
                bn * ht * wd * pol.long()
    values = th.ones_like(indices, dtype=dtype, device=device)
    representation.put_(indices, values, accumulate=True)
    representation = th.clamp(representation, min=0, max=count_cutoff)
    representation = th.sum(representation, dim=0)
    # print(representation.shape)
    representation = representation.to(th.uint8)
    # if not self.fastmode:
    #     representation = representation.to(th.uint8)
    return representation
class EventImageConvert:

    def __init__(self, width=346, height=260, interval=0.25, class_num=300, output_path=None) :
        self.width = width
        self.height = height
        
        self.interval = interval
        self.output_path = output_path

        self.class_num = class_num

        self.snn1 = snn.Leaky(beta=0.3, reset_mechanism="subtract")
        self.snn1 = self.snn1.to('cuda')

    def show_events(self,events):
        """
        plot events in three-dimensional space.

        Inputs:
        -------
            true events   - the true event signal.
            events   - events include true event signal and noise events.
            width    - the width of AER sensor.
            height   - the height of AER sensor.

        Outputs:
        ------
            figure     - a figure shows events in three-dimensional space.

        """
        ON_index = np.where(events['polarity'] == 1)
        OFF_index = np.where(events['polarity'] == 0)

        
        fig = plt.figure('{} * {}'.format(self.width, self.height))
        ax = fig.gca(projection='3d')

        ax.scatter(events['timestamp'][ON_index]-events['timestamp'][0], events['x'][ON_index], events['y'][ON_index], c='red', label='ON', s=3)  # events_ON[1][:]
        ax.scatter(events['timestamp'][OFF_index]-events['timestamp'][0], events['x'][OFF_index], events['y'][OFF_index], c='mediumblue', label='OFF', s=3)

        font1 = {'family': 'Times New Roman', 'size': 20}
        font1_x = {'family': 'Times New Roman', 'size': 19}
        font2 = {'size': 13}
        ax.set_xlabel('t(us)', font1_x)  # us 
        ax.set_ylabel('x', font1)
        ax.set_zlabel('y', font1)
        plt.show()

    def make_color_histo(self,events, img=None):
        """
        simple display function that shows negative events as blue dots and positive as red one
        on a white background
        args :
            - events structured numpy array: timestamp, x, y, polarity.
            - img (numpy array, height x width x 3) optional array to paint event on.
            - width int.
            - height int.
        return:
            - img numpy array, height x width x 3).
        """

        if img is None:
            img = 255 * np.ones((self.height, self.width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 255
        if events.size:
            
            assert events[:,0].max() < self.width, "out of bound events: x = {}, w = {}".format(events[:,0].max(), self.width)
            assert events[:,1].max() < self.height, "out of bound events: y = {}, h = {}".format(events[:,1].max(), self.height)

            ON_index = np.where(events[:,3] == 1)
            img[events[:,1][ON_index], events[:,0][ON_index], :] = [30, 30, 220] * events[:,3][ON_index][:, None]  # red
            OFF_index = np.where(events[:,3] == 0)
            img[events[:,1][OFF_index], events[:,0][OFF_index], :] = [200, 30, 30] * (events[:,3][OFF_index] + 1)[:,None]  # blue
        return img

    def _get_frames_NUM(self, input_filename):
        """
            Get the frame count for each event data
        """
        
        # with AedatFile(input_filename) as f:
        #     events = np.hstack([event for event in f['events'].numpy()])
        events_stream = np.load(input_filename)
        timestamps = events_stream['t']
        # print('p:', min(events_stream['p']))
            # timestamps = [t[0] for t in events]
        return math.ceil( ( max(timestamps) - min(timestamps) ) * 1e-6 / self.interval )

    # def _events_to_event_images(self,input_filename, output_file, relative_path, aps_frames_NUM, interval, label):
    def _events_to_event_images(self,input_filename, output_file, aps_frames_NUM, interval):
        """
        Mapping asynchronous events into event images
        args :
            - input_file:.aedat file, saving dvs events.
            - output_file: the output filename saving timestamps.
            - relative_path : relative path of png image with respect to the root path 
            - aps_frames_NUM: the number of the event data.
            - interval:time interval
            - label:the label of action
        return:
            - event_image
            - txt:saving the name of timestamp,frames_num,label
        """
        mem_dir = self.snn1.init_leaky()
        if os.path.exists(input_filename):
        
            # with AedatFile(input_filename) as f:
                # events = np.hstack([event for event in f['events'].numpy()])
            events_stream = np.load(input_filename)
            x, y, ts, pol = events_stream['x'], events_stream['y'], events_stream['t'], events_stream['p']
            # print((np.array([x,y,ts,pol]).shape))
            # print(max(np.array([x,y,ts,pol]).transpose()[0]))
            events = np.array([x,y,ts,pol]).transpose()

            start_timestamp = events[0][2] 

            # saving event images.
            for i in range(int(aps_frames_NUM)):     
                
                start_index = np.searchsorted(events[:,2], int(start_timestamp)+i*interval*1e6) 
                end_index = np.searchsorted(events[:,2], int(start_timestamp)+(i+1)*interval*1e6)

                print("start_index=",start_index)
                print("end_index=",end_index)

                rec_events = events[start_index:end_index]

                TCM = construct(torch.from_numpy(rec_events[:,0]), torch.from_numpy(rec_events[:,1]), torch.from_numpy(rec_events[:,3]), torch.from_numpy(rec_events[:,2]))
                TCM = TCM.to('cuda')
                output = []

                for j in range(TCM.shape[0]):
                    inp_img = TCM[j].float()
                    # Add two dimensions (batch size and channels)
                    inp_img = inp_img[None, None, :]
                    spk_dir, mem_dir = self.snn1(inp_img, mem_dir) 
                    output.append(spk_dir)
                output = torch.cat(output, dim=1)

                output = output.squeeze(0)


                TCM = output.detach().cpu().numpy()
                j_values = np.arange(1, TCM.shape[0] // 2 + 1)  # [1, 2, ..., N//2]
                multipliers = np.repeat(j_values, 2)  # [1, 1, 2, 2, ..., N//2, N//2]
                TCM = TCM * multipliers[:, np.newaxis, np.newaxis]  # 广播乘法

                TCM = np.sum(TCM, axis=0)
                TCM = TCM.astype(np.float64)
                TCM = np.divide(TCM, 
                        np.amax(TCM, axis=0, keepdims=True),
                        out=np.zeros_like(TCM),
                        where=TCM!=0)
                TCM = TCM * 255

                event_image = TCM
                save_path = output_file +'/{:08d}.png'.format(i)

                cv2.imwrite(save_path, event_image)

                # print('The filename {}, the {} frame has been done!'.format(input_filename, i+1))
                # output_filename.write(relative_path+ "_dvs" + " {}".format(aps_frames_NUM) + " {}".format(label) + '\n')  # save the timestamp,frames_NUM,label
        
    

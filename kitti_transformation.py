# -*- coding: utf-8 -*-
"""
Created on Fri May 24 02:02:39 2019

@author: Alice
"""


import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

# load mapping raw data names and indexes
# Change raw info format for using pykitty library
def reading_files(MAPPING_PATH=''):
    with open(os.path.join(MAPPING_PATH,'train_mapping.txt'),'r') as f:
        data=f.read()[:-1]
        raws = data.split('\n')
    raw_info=list(map(lambda x: x.split(' '), raws))
    date_drive_frame=list(map(lambda x: [x[0],x[1].split('_')[4],int(x[2])],raw_info))
    with open(os.path.join(MAPPING_PATH,'train_rand.txt'),'r') as f:
        data=f.read()[:-1]
        indexes = data.split(',')
        indexes=np.array(indexes,dtype = int)
    return indexes,date_drive_frame


#input: 
#delta - number of frames between neighbouring elements that can belong to one sequence, default = 1 
#threshold - length of sequence of frames to be fine for work

#output: sequence_dataloader - dictionary {number of sequence:{date,drive,list of couple frame-number of file in bird eye view,labels for each frame}}

#WARNING: change the order of frames opposite to timeline for the last frame in sequence to be in the (0,0,0) point
#as in sample we have to choose the frame and n previous ones (look the article "Adding Temporal Information" section)

def sequence_dict_creation(MAPPING_PATH='',delta = 1, threshold = 10,LABELS_PATH=''):
#date_drive_frame - list of lists with raw data info for each file in bird eye view dataset 
#(number of element = number of bin file)
    indexes,date_drive_frame = reading_files(MAPPING_PATH)
    
    sequences={}#{date: {drive: {frame: {number of sequence: array[number of frame,number of image in bird eye view dataset]}}}}

    #1)
    #initially create dictionary {date: {drive: {frame: array[number of frame,number of image in bird eye view dataset]}}}
    #without separation on several sequences depending on delta between frames numbers
    for i,info in enumerate(date_drive_frame):
        date = info[0]
        drive = info[1]
        frame = info[2]
        if date not in sequences.keys():
            sequences[date]={}
        if drive not in  sequences[date].keys():
            sequences[date][drive]=[]

        sequences[date][drive].append((int(frame),np.where(indexes == (i+1))[0].item()))

    seq_num = 0

    for date in sequences.keys():
        for drive in sequences[date].keys():
            sequences[date][drive] = np.array(sorted(sequences[date][drive],key = lambda x: x[0]))
            seq_num += 1

    print('Data consists of ',seq_num,' unfiltered sequences')
    
    #2)
    #we can check is there already existed sequences with deltas between frames less than some given delta
    #with delta = 10 there are 105 of 141 sequences

    def check_seq_delta(seq_array,delta = delta):
        it = iter(seq_array[:,0])
        first = next(it)
        return all(b - a < delta for a, b in enumerate(it, first + 1))

    #3)
    #separate sequence of frames if two neighbouring frames have delta more than given one

    def separate_sequences(seq_array,delta = delta,threshold = threshold):
        start = seq_array[:,0]
        end = start[1:]

        nums = [i+1 for i, (a, b) in enumerate(zip(start[:-1],end)) if b - a > delta]

        seq = {}

        if len(nums) == 0:
            if seq_array.shape[0]<threshold:
                return seq,0
            seq[0] = seq_array

            return seq,1

        if nums[0]!=0:
            nums.insert(0,0)
        nums.append(-1)

        j = 0
        for i in range(len(nums)-1):
            seq_prob = seq_array[nums[i]:nums[i+1],:]
            if seq_prob.shape[0]>threshold:
                seq[j] = seq_prob
                j+=1

        return seq, len(list(seq.keys()))

    seq_num = 0
    
    for date in sequences.keys():
        drives = list(sequences[date].keys())
        for drive in drives:
            seq, num=separate_sequences(sequences[date][drive],delta = delta, threshold = threshold)
            if num==0:
                del sequences[date][drive]
            else:
                seq_num +=num
                sequences[date][drive] = seq

    print('Final number of sequences ',seq_num)
    
    #extract labels of cars
    labels = []
    for i in range(len(indexes)):

        prefix = str.zfill(str(i),6)
        labelname = prefix+'.txt'
        with open(os.path.join(LABELS_PATH,labelname),'r') as f:
            data = f.read().split('\n')
            data=list(map(lambda x: x.split(' '),data))
            #ry,xmin,ymin,xmax,ymax,height,width,length,x,y,z,alpha
            
#             labels.append(np.array([info[3:] for info in data if info[0]=='Car'],dtype = float))
            labels.append(np.array([info[3:11]+[info[13],-float(info[11]),-float(info[12])]+[info[14]] for info in data if info[0]=='Car'],dtype=float))

    #renumber dictionary to format {sequence_number:{date:,drive:,[number_of_frame,number of bird eye view file]}}
    sequences_dataloader={}
    i = 0
    for date in sequences.keys():
        for drive in sequences[date].keys():
            for nums in sequences[date][drive].keys():
                seq_labels = [labels[num] for num in sequences[date][drive][nums][:,1]]
                sequences_dataloader[i]={'date':date,'drive':drive,'frame_file':sequences[date][drive][nums][::-1],'labels':seq_labels[::-1]}
                i+=1
    
    return sequences_dataloader

# Transform all the frame in sequence to one coordinate system
#input - date,drive,frames,labels info from sequences and folder with raw data
#output - list of transformed clouds, list of transformed labels in a format 
#[x,y,z(coordinate of object),x1,y1,...,x4,y4(coordinates of 2d boxes),height]
def coordinate_transform(date,drive,frames,labels,basedir=''):
    def label_transform(label,transform_matrix):
        def angle(v1, v2, acute=True):
        # v1 is your firsr vector
        # v2 is your second vector
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if (acute == True):
                return angle
            else:
                return 2 * np.pi - angle
            
        height, width, length = label[5:8]
        x,y,z = label[-4:-1]
        x+=0.27
        x+=0.81
        y+=0.32
        z+=(1.73-0.93)
        xt,yt,zt=(transform_matrix@np.array([x,y,z,1]))[:3]
        ry_old = float(label[-1])
        ry_old=-ry_old
        Rm = np.array([[np.cos(ry_old),np.sin(ry_old),0],[-np.sin(ry_old),np.cos(ry_old),0],[0,0,1]])
        box_vertices = [(width/2,length/2,0),(width/2,-length/2,0),(-width/2,-length/2,0),(-width/2,length/2,0)]
        box_vertices_transformed=[]
        for x0,y0,z0 in box_vertices:
            x0r,y0r,z0r=Rm@np.array([x0,y0,z0],dtype = float)+np.array([x,y,z],dtype = float)
    
            
            x0t,y0t,z0t=(transform_matrix@np.array([x0r,y0r,z0r,1]))[:3]#-np.array([xt,yt,zt])
            box_vertices_transformed+=[x0t,y0t]
        return np.array([xt,yt,zt]+box_vertices_transformed+[height],dtype = float)

    try:
        dataset = pykitti.raw(basedir, date, drive, frames=frames)
    except Exception:
        return None, None
    
    init_frame = dataset.get_velo(0)
    init_frame[:,0]+=0.81
    init_frame[:,1]+=0.32
    init_frame[:,2]+=(1.73-0.93)
    init_frame[:,3]= np.ones(init_frame.shape[0])
    init_frame=(dataset.oxts[0].T_w_imu@init_frame.T).T
    output = [init_frame[:,:3]]
    output_labels = []
    if len(labels[0])!=0:
        init_label_coordinates=labels[0][:,-4:-1]
        init_label_coordinates=np.hstack([init_label_coordinates,np.ones((init_label_coordinates.shape[0],1))])
        init_label_coordinates = (dataset.oxts[0].T_w_imu@init_label_coordinates.T).T
        rys=np.array([label_transform(label,dataset.oxts[0].T_w_imu) for label in labels[0] ],dtype = float)
        
        output_labels = [rys]
        
        
    else:
        output_labels.append(np.array([]))
        
    for i in range(1,len(frames)):
        frame = dataset.get_velo(i)
        frame[:,0]+=0.81
        frame[:,1]+=0.32
        frame[:,2]+=(1.73-0.93)
        frame[:,3]= np.ones(frame.shape[0])#replace color with ones for transformation
        #transformation
        frame=(dataset.oxts[i].T_w_imu@frame.T).T
        output.append(frame[:,:3])
        if len(labels[i])!=0:
            label_coordinates = labels[i][:,-4:-1]
            label_coordinates=np.hstack([label_coordinates,np.ones((label_coordinates.shape[0],1))])
            label_coordinates = (dataset.oxts[i].T_w_imu@label_coordinates.T).T
            rys=np.array([label_transform(label,dataset.oxts[i].T_w_imu) for label in labels[i] ],dtype = float)
            output_labels.append(rys)
            
        else:
            output_labels.append(np.array([]))
    return output, output_labels   

    
#plot !transformed! labels
def plot_targets(targets,angles=(0,90)):
    f = plt.figure(figsize = (20,20))
    ax = f.add_subplot(111, projection='3d')
    colors=['red','blue','orange','magenta','green','yellow','grey','cyan']
    coords=[]
    for i,target in enumerate(targets): 
        c=colors[i%(len(colors))]
        for info in target:
            
            x,y,z = info[:3]
            coords.append([x,y,z])
            ax.scatter(x,y,z, s=100,c=c)
            boxes=info[3:-1].reshape(-1,2)
            for k in range(len(boxes)):
                x0,y0=boxes[k-1]
                x1,y1=boxes[k]
                ax.plot([x0,x1],[y0,y1],[z,z],c='black',lw=1)
    coords = np.array(coords)
    ax.auto_scale_xyz([coords[:,:2].min(),coords[:,:2].max()], [coords[:,:2].min(),coords[:,:2].max()], [coords[:,2].min(),coords[:,2].max()])
    ax.view_init(angles[0], angles[1]) 
    plt.show()
    
#input - list of transformed clouds, list of !transformed! targets
#input - list of clouds, list of targets

def plot_clouds(clouds, targets,delta = 10,angles=(90,180),xlim=(-10,10),ylim=(-10,10)):
    
    
    
    f = plt.figure(figsize = (20,20))
    ax = f.add_subplot(111, projection='3d')
    colors=['red','blue','orange','magenta','green','yellow','grey','cyan']
    
    init_velo = clouds[0]
    
    #plot only each delta_th points
    velo_range = range(0, init_velo.shape[0], delta)
    ax.scatter(init_velo[velo_range, 0],
                init_velo[velo_range, 1],
                init_velo[velo_range, 2],s=1,c=colors[0])
    for i,cloud in enumerate(clouds[1:]):
        velo_range = range(0, cloud.shape[0], delta)
        ax.scatter(cloud[velo_range, 0],
                cloud[velo_range, 1],
                cloud[velo_range, 2],s=1,c=colors[(i+1)%len(colors)])
    
    for i,target in enumerate(targets): 
        c=colors[i%len(colors)]
        for m,info in enumerate(target):
            
            x,y,z = info[:3]
            ax.scatter(x,y,z, s=200,c=c)
            boxes=info[3:-1].reshape(-1,2)
            for i in range(len(boxes)):
                x0,y0=boxes[i-1]
                x1,y1=boxes[i]
                ax.plot([x0,x1],[y0,y1],[z,z],c=c,lw=1)

    ax.view_init(angles[0], angles[1])
    ax.set_xlim3d(xlim[0], xlim[1])
    ax.set_ylim3d(ylim[0], ylim[1])
    plt.show()


#transformation pipeline
#input: sequences - dictionary, output of "sequence_dict_creation" function
#number of sequence - element from sequences.keys()
#start_frame_num,end_frame_num(not including last element) - numbers of elements in frame lists from 0 to len(sequence)
def transform_kitti(sequences,number_of_sequence,start_frame_num,end_frame_num,RAW_DATA_PATH = ''):
    date = sequences[number_of_sequence]['date']
    drive = sequences[number_of_sequence]['drive']
    frames = sequences[number_of_sequence]['frame_file'][start_frame_num:end_frame_num,0]
    labels = sequences[number_of_sequence]['labels'][start_frame_num:end_frame_num]
    transformed_clouds, transformed_labels = coordinate_transform(date,drive,frames,labels,basedir=RAW_DATA_PATH)
    return transformed_clouds, transformed_labels


def transform_kitti_full(sequences,number_of_sequence,RAW_DATA_PATH = ''):
    date = sequences[number_of_sequence]['date']
    drive = sequences[number_of_sequence]['drive']
    frames = sequences[number_of_sequence]['frame_file'][:,0]
    labels = sequences[number_of_sequence]['labels']
    transformed_clouds, transformed_labels = coordinate_transform(date,drive,frames,labels,basedir=RAW_DATA_PATH)
    return transformed_clouds, transformed_labels

#dataset for pytorch
class KittiDataset(Dataset):
    def __init__(self, n, indexes, sequences, RAW_DATA_PATH=""):
        """
        Args:
            n (int): Number of frames to take
            indexes (list int): List of number of sequence to work with
            sequences (dictionary): output of function sequence_dict_creation
        """
        
        self.sequences = sequences
        self.RAW_DATA_PATH = RAW_DATA_PATH
        self.n = n
        self.indexes=indexes
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        start = np.random.randint(0,len(list(self.sequences[index]['labels']))-self.n)
        transformed_clouds, transformed_labels = transform_kitti(self.sequences,
                                                                index,
                                                                start_frame_num = start,
                                                                end_frame_num = start+self.n,
                                                                RAW_DATA_PATH = self.RAW_DATA_PATH)
        if transformed_clouds is None:
            return None, None
        
        transformed_clouds = list(map(lambda x: torch.from_numpy(x),transformed_clouds))
        transformed_labels = list(map(lambda x: torch.from_numpy(x),transformed_labels))

        return transformed_clouds, transformed_labels
    
#dataset for pytorch
class KittiDatasetFull(Dataset):
    def __init__(self, n, indexes, sequences, RAW_DATA_PATH=""):
        """
        Args:
            n (int): Number of frames to take
            indexes (list int): List of number of sequence to work with
            sequences (dictionary): output of function sequence_dict_creation
        """
        
        self.sequences = sequences
        self.RAW_DATA_PATH = RAW_DATA_PATH
        self.n = n
        self.indexes=indexes
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        transformed_clouds, transformed_labels = transform_kitti_full(self.sequences,
                                                                index,
                                                                RAW_DATA_PATH = self.RAW_DATA_PATH)
        if transformed_clouds is None:
            return None, None
        
        transformed_clouds = list(map(lambda x: torch.from_numpy(x),transformed_clouds))
        transformed_labels = list(map(lambda x: torch.from_numpy(x),transformed_labels))

        return transformed_clouds, transformed_labels
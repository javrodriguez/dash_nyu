#from datagenerator import DataGenerator
from csvreader import readcsv, writecsv
import numpy as np
import torch
from math import ceil
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from figure_saver import save_figure
import random

#print("Using {} threads datahandler".format(torch.get_num_threads()))

class DataHandler:

    def __init__(self, data_np, data_pt, time_np, time_pt, dim, ntraj, val_split, device, normalize, batch_type, batch_time, batch_time_frac, data_np_0noise, data_pt_0noise, img_save_dir, init_bias_y, my_fixed_val_for_hema = 0):
        self.data_np = data_np
        self.data_pt = data_pt
        self.data_np_0noise = data_np_0noise
        self.data_pt_0noise = data_pt_0noise
        self.time_np = time_np
        self.time_pt = time_pt
        self.dim = dim
        self.ntraj = ntraj
        self.batch_type = batch_type
        self.batch_time = batch_time
        self.batch_npoints = int(ceil(batch_time_frac * batch_time))
        self.my_fixed_val_for_hema = my_fixed_val_for_hema
        if normalize:
            self._normalize()
        self.device = device
        self.val_split = val_split
        self.epoch_done = False
        self.img_save_dir = img_save_dir
        self.init_bias_y = init_bias_y
        self.num_trajs_to_plot = 2
        #self.noise = noise

        self._calc_datasize()
        if batch_type == 'single':
            self._split_data_single(val_split)
            self._create_validation_set_single()
        elif batch_type == 'trajectory':
            self._split_data_traj(val_split)
            self._create_validation_set_traj()
            #self.compare_train_val_plot() #only plot it trajectory
        elif batch_type == 'batch_time':
            self._split_data_time(val_split)
            self._create_validation_set_time()
        else:
            print("Invalid batch type: '{}'".format(batch_type))
            raise ValueError
        

    @classmethod
    def fromcsv(cls, fp, device, val_split, normalize=False, batch_type='single', batch_time=1, batch_time_frac=1.0, noise = 0, img_save_dir = "", scale_expression = 1, log_scale = False, init_bias_y = 0, my_fixed_val_for_hema = 0):
        ''' Create a datahandler from a CSV file '''
        data_np, data_pt, t_np, t_pt, dim, ntraj, data_np_0noise, data_pt_0noise = readcsv(fp, device, noise_to_add = noise, scale_expression = scale_expression, log_scale = log_scale)
        return DataHandler(data_np, data_pt, t_np, t_pt, dim, ntraj, val_split, device, normalize, batch_type, batch_time, batch_time_frac, data_np_0noise, data_pt_0noise, img_save_dir, init_bias_y,  my_fixed_val_for_hema)

    @classmethod
    def fromgenerator(cls, generator, val_split, device, normalize=False):
        ''' Create a datahandler from a data generator '''
        data_np, data_pt, t_np, t_pt = generator.generate()
        return DataHandler(data_np, data_pt, t_np, t_pt, generator.dim, generator.ntraj, val_split, device, normalize)

    def saveascsv(self, fp):
        ''' Saves the data to a CSV file '''
        writecsv(fp, self.dim, self.ntraj, self.data_np, self.time_np)

    def _normalize(self):
        max_val = 0
        for data in self.data_pt:
            if torch.max(torch.abs(data)) > max_val:
                max_val = torch.max(torch.abs(data))
        for i in range(self.ntraj):
            self.data_pt[i] = torch.div(self.data_pt[i], max_val)
            self.data_np[i] = self.data_np[i] / max_val.numpy()

    def reset_epoch(self):
        self.train_set = self.train_set_original.copy()
        self.epoch_done = False

    def get_batch(self, batch_size):
        if self.batch_type == 'single':
            return self._get_batch_single(batch_size)
        elif self.batch_type == 'trajectory':
            return self._get_batch_traj(batch_size)
        elif self.batch_type == 'batch_time':
            return self._get_batch_time(batch_size)

    def _get_batch_single(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > batch_size:
            indx = np.random.choice(train_set_size, batch_size, replace=False)
        else:
            indx = np.arange(train_set_size)
            self.epoch_done = True # We are doing the last items in current epoch
        indx = np.sort(indx)[::-1]
        batch_indx = [self.train_set[x] for x in indx]
        batch = []
        t = []
        target = []
        for i in batch_indx:
            potential_target = self.data_pt[i[0]][i[1] + 1]
            if not torch.isnan(potential_target).all().item():
                batch.append(self.data_pt[i[0]][i[1]])
                target.append(potential_target)
                t.append(torch.stack([self.time_pt[i[0]][i[1] + ii] for ii in range(2)]))
        for i in indx:
            self.train_set.pop(i)
        # Convert the lists to tensors
        reshape_size = batch_size if train_set_size > batch_size else train_set_size
        batch = torch.stack(batch).to(self.device)
        t = torch.stack(t).to(self.device)
        target = torch.stack(target).to(self.device)
        return batch, t, target

    def _get_batch_traj(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > 1:
            i = np.random.choice(train_set_size, replace=False)
            indx = self.train_set[i]
            self.train_set = np.delete(self.train_set, i)
        else:
            indx = self.train_set[0]
            self.epoch_done = True # We are doing the last items in current epoch
        # Convert the lists to tensors
        batch = self.data_pt[indx][0:-1].to(self.device)
        t = []
        for i in range(self.time_pt[indx].shape[0] - 1):
            t.append(torch.tensor([self.time_pt[indx][i], self.time_pt[indx][i+1]]))
        t = torch.stack(t)
        target = self.data_pt[indx][1::].to(self.device)

        #IH: 9/10/2021 - added these to handle unequal time availability 
        #comment these out when not requiring nan-value checking
        not_nan_idx = [i for i in range(len(t)) if not torch.isnan(t[i]).any().item()]
        batch = batch[not_nan_idx]
        t = t[not_nan_idx]
        target = target[not_nan_idx]


        return batch, t, target

    def _get_batch_time(self, batch_size):
        ''' Get a batch of data, it's corresponding time data and the target data '''
        train_set_size = len(self.train_set)

        if train_set_size > batch_size:
            indx = np.random.choice(train_set_size, batch_size, replace=False)
        else:
            indx = np.arange(train_set_size)
            self.epoch_done = True # We are doing the last items in current epoch
        indx = np.sort(indx)[::-1]
        batch_indx = [self.train_set[x] for x in indx]
        batch = []
        t = []
        target = []
        for i in batch_indx:
            sub_indx = np.random.choice(np.arange(start=i[1], stop=i[1]+self.batch_time), size=self.batch_npoints, replace=False)
            sub_indx = np.sort(sub_indx)
            sub_indx = np.append(sub_indx, sub_indx[-1]+1)
            batch.append(torch.stack([self.data_pt[i[0]][ii] for ii in sub_indx[0:-1]]))
            target.append(torch.stack([self.data_pt[i[0]][ii+1] for ii in sub_indx[1::]]))
            t.append(torch.tensor([self.time_pt[i[0]][ii] for ii in sub_indx]))
        for i in indx:
            self.train_set.pop(i)
        # Convert the lists to tensors
        reshape_size = batch_size if train_set_size > batch_size else train_set_size
        batch = torch.stack(batch).squeeze().to(self.device)
        t = torch.stack(t).to(self.device)
        target = torch.stack(target).squeeze().to(self.device)
        return batch, t, target

    def _split_data_single(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int((self.datasize - self.ntraj) * val_split)
        all_indx = np.arange(len(self.indx))
        val_indx = np.random.choice(all_indx, size=self.n_val, replace=False)
        
        #val_indx = np.array([ 16,  39,  42,  51,  55,  68,  78, 101, 107, 144, 160, 184, 208,233, 237, 318, 319, 320, 324, 335, 378, 393, 394, 422, 433, 434, 447, 469, 482, 491, 493, 495, 513, 529, 541, 546, 563, 570, 577])
        train_indx = np.setdiff1d(all_indx, val_indx, assume_unique=True)
        self.val_set_indx = [self.indx[x] for x in val_indx]
        self.train_set_original = [self.indx[x] for x in train_indx]
        self.train_data_length = len(self.train_set_original)
      

    def _split_data_traj(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int(round(self.ntraj * val_split))
        all_indx = np.arange(self.ntraj)
        val_indx = np.random.choice(all_indx, size=self.n_val, replace=False)
        train_indx = np.setdiff1d(all_indx, val_indx, assume_unique=True)
        self.val_set_indx = val_indx
        self.train_set_original = train_indx
        self.train_data_length = len(self.train_set_original)

    def _split_data_time(self, val_split):
        ''' Split the data into a training set and validation set '''
        self.n_val = int((self.datasize - self.ntraj) * val_split)
        all_indx = np.arange(len(self.indx))
        val_indx = np.random.choice(all_indx, size=self.n_val, replace=False)
        train_indx = np.setdiff1d(all_indx, val_indx, assume_unique=True)
        self.val_set_indx = [self.indx[x] for x in val_indx]
        self.train_set_original = [self.indx[x] for x in train_indx]
        self.train_data_length = len(self.train_set_original)

    def _calc_datasize(self):
        if self.batch_type == 'single':
            self._calc_datasize_single()
        elif self.batch_type == 'trajectory':
            self.datasize = self.ntraj
        elif self.batch_type == 'batch_time':
            self._calc_datasize_time()

    def _calc_datasize_single(self):
        ''' Calculate the total number of data points '''
        self.indx = []
        for i in range(self.ntraj):
            for j in range(self.data_pt[i].shape[0] - 1):
                potential_target = self.data_pt[i][j + 1]
                if not torch.isnan(potential_target).all().item():
                    self.indx.append([i, j])
        self.datasize = len(self.indx)

    def _calc_datasize_time(self):
        ''' Calculate the total number of data points '''
        self.indx = []
        for i in range(self.ntraj):
            for j in range(self.data_pt[i].shape[0] - self.batch_time):
                self.indx.append([i, j])
        self.datasize = len(self.indx)

    def get_mu0(self):
        ''' Get the initial conditions '''
        return self.data_pt[0][0]

    def get_true_mu_set_pairwise(self, val_only = False, batch_type = "trajectory"):
        ''' Get the true mu set for pairwise comparison '''
        if val_only:
            if batch_type == "trajectory":
                data_pw = []
                t_pw = []
                target_pw = []
                for i in self.val_set_indx:
                    data_pw.append(self.data_pt[i][0:-1])
                    t_pw.append(torch.tensor([self.time_pt[i][0], self.time_pt[i][-1]]))
                    target_pw.append(self.data_pt[i][-1])
                data_pw = torch.stack(data_pw).to(self.device)
                t_pw = torch.stack(t_pw).to(self.device)
                target_pw = torch.stack(target_pw).to(self.device)
                return data_pw, t_pw, target_pw
            else:
                data_pw = []
                t_pw = []
                target_pw = []
                for i in self.val_set_indx:
                    for j in range(self.data_pt[i[0]].shape[0] - 1):
                        potential_target = self.data_pt[i[0]][j + 1]
                        if not torch.isnan(potential_target).all().item():
                            data_pw.append(self.data_pt[i[0]][j])
                            target_pw.append(potential_target)
                            t_pw.append(torch.stack([self.time_pt[i[0]][j], self.time_pt[i[0]][j + 1]]))
                data_pw = torch.stack(data_pw).to(self.device)
                t_pw = torch.stack(t_pw).to(self.device)
                target_pw = torch.stack(target_pw).to(self.device)
                return data_pw, t_pw, target_pw
        else:
            if batch_type == "trajectory":
                data_pw = []
                t_pw = []
                target_pw = []
                for i in range(self.ntraj):
                    data_pw.append(self.data_pt[i][0:-1])
                    t_pw.append(torch.tensor([self.time_pt[i][0], self.time_pt[i][-1]]))
                    target_pw.append(self.data_pt[i][-1])
                data_pw = torch.stack(data_pw).to(self.device)
                t_pw = torch.stack(t_pw).to(self.device)
                target_pw = torch.stack(target_pw).to(self.device)
                return data_pw, t_pw, target_pw
            else:
                data_pw = []
                t_pw = []
                target_pw = []
                for i in range(self.ntraj):
                    for j in range(self.data_pt[i].shape[0] - 1):
                        potential_target = self.data_pt[i][j + 1]
                        if not torch.isnan(potential_target).all().item():
                            data_pw.append(self.data_pt[i][j])
                            target_pw.append(potential_target)
                            t_pw.append(torch.stack([self.time_pt[i][j], self.time_pt[i][j + 1]]))
                data_pw = torch.stack(data_pw).to(self.device)
                t_pw = torch.stack(t_pw).to(self.device)
                target_pw = torch.stack(target_pw).to(self.device)
                return data_pw, t_pw, target_pw

    def get_true_mu_set_init_val_based(self, val_only = False): 
        ''' Get the true mu set for initial value based comparison '''
        if val_only:
            data = []
            t = []
            target = []
            for i in self.val_set_indx:
                data.append(self.data_pt[i][0])
                t.append(self.time_pt[i])
                target.append(self.data_pt[i][1::])
            data = torch.stack(data).to(self.device)
            t = torch.stack(t).to(self.device)
            target = torch.stack(target).to(self.device)
            return data, t, target
        else:
            data = []
            t = []
            target = []
            for i in range(self.ntraj):
                data.append(self.data_pt[i][0])
                t.append(self.time_pt[i])
                target.append(self.data_pt[i][1::])
            data = torch.stack(data).to(self.device)
            t = torch.stack(t).to(self.device)
            target = torch.stack(target).to(self.device)
            return data, t, target

    def get_times(self):
        ''' Get the time points '''
        return self.time_np

    def calculate_trajectory(self, odenet, method, num_val_trajs, fixed_traj_idx = None):
        #print(self.val_set_indx)
        #print(num_val_trajs)
        if fixed_traj_idx is not None:
            traj_indx = [fixed_traj_idx]
        else:
            if self.n_val > 0:
                traj_indx = self.val_set_indx[:num_val_trajs]
            else:
                traj_indx = np.random.choice(self.ntraj, num_val_trajs, replace=False)
        
        trajectories = []
        all_plotted_samples = []
        extrap_timepoints = []
        
        for i in traj_indx:
            all_plotted_samples.append(i)
            # Get the initial condition
            y0 = self.data_pt[i][0]
            # Get the time points
            t = self.time_pt[i]
            # Calculate the trajectory
            with torch.no_grad():
                trajectory = odeint(odenet, y0, t, method=method)
            trajectories.append(trajectory)
            extrap_timepoints.append(t)
        
        return trajectories, all_plotted_samples, extrap_timepoints

    def calculate_trajectory_pathreg(self, pathreg_model, method, num_val_trajs, fixed_traj_idx = None):
        #print(self.val_set_indx)
        #print(num_val_trajs)
        if fixed_traj_idx is not None:
            traj_indx = [fixed_traj_idx]
        else:
            if self.n_val > 0:
                traj_indx = self.val_set_indx[:num_val_trajs]
            else:
                traj_indx = np.random.choice(self.ntraj, num_val_trajs, replace=False)
        
        trajectories = []
        all_plotted_samples = []
        extrap_timepoints = []
        
        for i in traj_indx:
            all_plotted_samples.append(i)
            # Get the initial condition
            y0 = self.data_pt[i][0]
            # Get the time points
            t = self.time_pt[i]
            # Calculate the trajectory
            with torch.no_grad():
                trajectory = pathreg_model.predict_trajectory(y0, t, method=method)
            trajectories.append(trajectory)
            extrap_timepoints.append(t)
        
        return trajectories, all_plotted_samples, extrap_timepoints

    def _create_validation_set_single(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_time = []
        self.val_target = []
        for i in self.val_set_indx:
            for j in range(self.data_pt[i[0]].shape[0] - 1):
                potential_target = self.data_pt[i[0]][j + 1]
                if not torch.isnan(potential_target).all().item():
                    self.val_data.append(self.data_pt[i[0]][j])
                    self.val_target.append(potential_target)
                    self.val_time.append(torch.stack([self.time_pt[i[0]][j], self.time_pt[i[0]][j + 1]]))
        self.val_data = torch.stack(self.val_data).to(self.device)
        self.val_time = torch.stack(self.val_time).to(self.device)
        self.val_target = torch.stack(self.val_target).to(self.device)

    def _create_validation_set_traj(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_time = []
        self.val_target = []
        for i in self.val_set_indx:
            self.val_data.append(self.data_pt[i][0:-1])
            t = []
            for j in range(self.time_pt[i].shape[0] - 1):
                t.append(torch.tensor([self.time_pt[i][j], self.time_pt[i][j+1]]))
            t = torch.stack(t)
            self.val_time.append(t)
            self.val_target.append(self.data_pt[i][1::])
        self.val_data = torch.stack(self.val_data).to(self.device)
        self.val_time = torch.stack(self.val_time).to(self.device)
        self.val_target = torch.stack(self.val_target).to(self.device)

    def _create_validation_set_time(self):
        ''' Create the validation set '''
        self.val_data = []
        self.val_time = []
        self.val_target = []
        for i in self.val_set_indx:
            for j in range(self.data_pt[i[0]].shape[0] - self.batch_time):
                sub_indx = np.arange(start=j, stop=j+self.batch_time)
                sub_indx = np.append(sub_indx, sub_indx[-1]+1)
                self.val_data.append(torch.stack([self.data_pt[i[0]][ii] for ii in sub_indx[0:-1]]))
                self.val_target.append(torch.stack([self.data_pt[i[0]][ii+1] for ii in sub_indx[1::]]))
                self.val_time.append(torch.tensor([self.time_pt[i[0]][ii] for ii in sub_indx]))
        self.val_data = torch.stack(self.val_data).squeeze().to(self.device)
        self.val_time = torch.stack(self.val_time).to(self.device)
        self.val_target = torch.stack(self.val_target).squeeze().to(self.device)

    def get_validation_set(self):
        ''' Get the validation set '''
        return self.val_data, self.val_time, self.val_target, self.n_val

    def compare_train_val_plot(self):
        ''' Plot the training and validation sets '''
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(self.train_data[:, 0], self.train_data[:, 1], c='blue', label='Training')
        ax.scatter(self.val_data[:, 0], self.val_data[:, 1], c='red', label='Validation')
        ax.legend()
        plt.savefig('{}/train_val_split.png'.format(self.img_save_dir))
        plt.close() 
import json
import os
import os.path

import numpy as np

import torch

from torch.utils.data import Dataset
from copy import deepcopy
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml
import copy

MOTION_PATH = './mocap/CMU/'
class Dataset(Dataset):
    def __init__(self, mode='train', past_len=10, future_len=25, sample_rate=1):
        data_name = os.listdir(MOTION_PATH)
        self.past_len = past_len
        self.future_len = future_len
        self.data = []
        self.idx2frame_two = [] # (seq_id, sub_seq_id, bias)
        self.idx2frame_one = []
        framerate = 120
        self.two_person_data = []
        self.one_person_data = []
        num = 0
        for ii in range(3):
            if ii==0:
                A='18_19_Justin/'
                B='18_19_rory/'
            if ii==1:
                A='20_21_Justin1/'
                B='20_21_rory1/'
            if ii==2:
                A='22_23_justin/'
                B='22_23_Rory/'

            if ii==1 and mode == 'train':
                continue
            elif ii!=1 and mode == 'test':
                continue

            poses_path_A = MOTION_PATH+A
            poses_path_B = MOTION_PATH+B
            
            for iii, each in enumerate(os.listdir(poses_path_A)):
                if each[:2] != A[:2]:
                    continue
                if each not in os.listdir(poses_path_B):
                    continue
                # print(each)
                bdata_A = np.load(poses_path_A + each)
                bdata_B = np.load(poses_path_B + each)
                if bdata_A['mocap_framerate'] != framerate or bdata_B['mocap_framerate'] != framerate:
                    sr = int(sample_rate * bdata_A['mocap_framerate'] // framerate)
                    if sr < 1:
                        raise Exception('sample rate less than 1')
                else:
                    sr = sample_rate
                two_person_poses = torch.cat([torch.from_numpy(bdata_A['poses']).unsqueeze(0), torch.from_numpy(bdata_B['poses']).unsqueeze(0)])
                two_person_trans = torch.cat([torch.from_numpy(bdata_A['trans']).unsqueeze(0), torch.from_numpy(bdata_B['trans']).unsqueeze(0)])
                two_person_betas = torch.cat([torch.from_numpy(bdata_A['betas']).unsqueeze(0), torch.from_numpy(bdata_B['betas']).unsqueeze(0)])
                two_person_gender = [str(bdata_A['gender']), str(bdata_B['gender'])]
                frame_times = bdata_A['trans'].shape[0]
                two_person = {'length': frame_times,
                                   'trans': two_person_trans,
                                   'poses': two_person_poses,
                                   'betas': two_person_betas,
                                   'gender': two_person_gender,
                                   }
                fragment = (past_len + future_len) * sr
                for i in range(frame_times // fragment):
                    if mode == "test":
                        self.idx2frame_two.append((num, i * fragment, 1, sr))
                    elif i == frame_times // fragment - 1:
                        self.idx2frame_two.append((num, i * fragment, frame_times + 1 - (frame_times // fragment) * fragment, sr))
                    else:
                        self.idx2frame_two.append((num, i * fragment, fragment, sr))
                # Data keys available:['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']
                self.two_person_data.append(two_person)
                num += 1
        num = 0
        for k, name in tqdm(enumerate(data_name)):
            
            if len(name) > 3:
                continue
            
            if k % 4 == 0 and mode == 'train':
                continue
            elif k % 4 != 0 and mode == 'test':
                continue

            poses_path = MOTION_PATH+name
            
            for iii, each in enumerate(os.listdir(poses_path)):
                if each[:2] != name[:2]:
                    continue
                # print(each)
                bdata = np.load(poses_path + '/' + each)
                if bdata['mocap_framerate'] != framerate:
                    sr = int(sample_rate * bdata['mocap_framerate'] // framerate)
                    if sr < 1:
                        raise Exception('sample rate less than 1')
                else:
                    sr = sample_rate
                one_person_poses = torch.from_numpy(bdata['poses']).unsqueeze(0)
                one_person_trans = torch.from_numpy(bdata['trans']).unsqueeze(0)
                one_person_betas = torch.from_numpy(bdata['betas']).unsqueeze(0)
                one_person_gender = [str(bdata['gender'])]
                frame_times = bdata['trans'].shape[0]
                one_person = {'length': frame_times,
                                   'trans': one_person_trans,
                                   'poses': one_person_poses,
                                   'betas': one_person_betas,
                                   'gender': one_person_gender,
                                   }
                fragment = (past_len + future_len) * sr
                for i in range(frame_times // fragment):
                    if mode == "test":
                        self.idx2frame_one.append((num, i * fragment, 1, sr))
                    elif i == frame_times // fragment - 1:
                        self.idx2frame_one.append((num, i * fragment, frame_times + 1 - (frame_times // fragment) * fragment, sr))
                    else:
                        self.idx2frame_one.append((num, i * fragment, fragment, sr))
                self.one_person_data.append(one_person)
                num += 1
        self.mode = mode
        self.pose_dim = one_person['poses'].shape[-1]

    def __getitem__(self, idx):
        if idx < len(self.idx2frame_two):
            index, frame_idx, bias, sr = self.idx2frame_two[idx]
            two_person = copy.deepcopy(self.two_person_data[index])
            two_person_start_frame = np.random.choice(bias) + frame_idx
            two_person_end_frame = two_person_start_frame + (self.past_len + self.future_len) * sr
            if self.mode == 'test':
                one_idx = idx % len(self.idx2frame_one)
            else:
                one_idx = np.random.choice(len(self.idx2frame_one))
            index_one, frame_idx_one, bias_one, sr_one = self.idx2frame_one[one_idx]
            one_person = copy.deepcopy(self.one_person_data[index_one])
            one_person_start_frame = np.random.choice(bias_one) + frame_idx_one
            one_person_end_frame = one_person_start_frame + (self.past_len + self.future_len) * sr_one
            two_person_trans = two_person['trans'][:, two_person_start_frame:two_person_end_frame]
            one_person_trans = one_person['trans'][:, one_person_start_frame:one_person_end_frame]
            # print(two_person_trans[:, 0], one_person_trans[:, 0])
            if self.mode == 'test':
                rand_range = idx % 2 + 1
                rand_value_0 = idx % 3
                rand_value_1 = idx % 5
                rand_value_2 = idx % 4
            else:
                rand_range = torch.rand(1)[0] * 2 + 1
                rand_value_0 = torch.rand(1)[0] * 2
                rand_value_1 = torch.rand(1)[0] * 4
                rand_value_2 = torch.rand(1)[0] * 3
        
            if rand_value_0 >= 1:
                if rand_value_1 > 2:
                    one_person['trans'][:, :, 0] -= min(one_person_trans[:, :, 0].min() - two_person_trans[:, :, 0].max(), 0) - rand_range
                else:
                    one_person['trans'][:, :, 0] -= max(one_person_trans[:, :, 0].max() - two_person_trans[:, :, 0].min(), 0) + rand_range
            if rand_value_0 <= 1.5:
                if rand_value_2 > 1.5:
                    one_person['trans'][:, :, 1] -= min(one_person_trans[:, :, 1].min() - two_person_trans[:, :, 1].max(), 0) - rand_range
                else:
                    one_person['trans'][:, :, 1] -= max(one_person_trans[:, :, 1].max() - two_person_trans[:, :, 1].min(), 0) + rand_range

            three_person_trans = torch.cat([two_person['trans'][:, two_person_start_frame:two_person_end_frame:sr], one_person['trans'][:, one_person_start_frame:one_person_end_frame:sr_one]])
            three_person_poses = torch.cat([two_person['poses'][:, two_person_start_frame:two_person_end_frame:sr], one_person['poses'][:, one_person_start_frame:one_person_end_frame:sr_one]])
            three_person_betas = torch.cat([two_person['betas'], one_person['betas']])
            three_person_gender = two_person['gender'] + one_person['gender']
            three_person = {
                'length': self.past_len + self.future_len,
                'trans': three_person_trans,
                'poses': three_person_poses,
                'betas': three_person_betas,
                'gender': three_person_gender,
                'global': 1,
            }

        else:
            if self.mode == 'test':
                rand_value = 0
            else:
                rand_value = torch.rand(1)[0]

            if rand_value < 0.5:
                index_one, frame_idx_one, bias_one, sr_one = self.idx2frame_one[idx - len(self.idx2frame_two)]
                one_person = copy.deepcopy(self.one_person_data[index_one])
                one_person_start_frame = np.random.choice(bias_one) + frame_idx_one
                one_person_end_frame = one_person_start_frame + (self.past_len + self.future_len) * sr_one
                if self.mode == 'test':
                    two_idx = idx % len(self.idx2frame_two)
                else:
                    two_idx = np.random.choice(len(self.idx2frame_two))
                index, frame_idx, bias, sr = self.idx2frame_two[two_idx]
                two_person = copy.deepcopy(self.two_person_data[index])
                two_person_start_frame = np.random.choice(bias) + frame_idx
                two_person_end_frame = two_person_start_frame + (self.past_len + self.future_len) * sr

                two_person_trans = two_person['trans'][:, two_person_start_frame:two_person_end_frame]
                one_person_trans = one_person['trans'][:, one_person_start_frame:one_person_end_frame]


                if self.mode == 'test':
                    rand_range = idx % 3 + 1
                    rand_value_0 = idx % 3
                    rand_value_1 = idx % 5
                    rand_value_2 = idx % 4
                else:
                    rand_range = torch.rand(1)[0] * 2 + 1
                    rand_value_0 = torch.rand(1)[0] * 2
                    rand_value_1 = torch.rand(1)[0] * 4
                    rand_value_2 = torch.rand(1)[0] * 3
            
                if rand_value_0 >= 1:
                    if rand_value_1 > 2:
                        one_person['trans'][:, :, 0] -= min(one_person_trans[:, :, 0].min() - two_person_trans[:, :, 0].max(), 0) - rand_range
                    else:
                        one_person['trans'][:, :, 0] -= max(one_person_trans[:, :, 0].max() - two_person_trans[:, :, 0].min(), 0) + rand_range
                if rand_value_0 <= 1.5:
                    if rand_value_2 > 1.5:
                        one_person['trans'][:, :, 1] -= min(one_person_trans[:, :, 1].min() - two_person_trans[:, :, 1].max(), 0) - rand_range
                    else:
                        one_person['trans'][:, :, 1] -= max(one_person_trans[:, :, 1].max() - two_person_trans[:, :, 1].min(), 0) + rand_range
                
                three_person_trans = torch.cat([one_person['trans'][:, one_person_start_frame:one_person_end_frame:sr_one], two_person['trans'][:, two_person_start_frame:two_person_end_frame:sr]])
                three_person_poses = torch.cat([one_person['poses'][:, one_person_start_frame:one_person_end_frame:sr_one], two_person['poses'][:, two_person_start_frame:two_person_end_frame:sr]])
                three_person_betas = torch.cat([one_person['betas'], two_person['betas']])
                three_person_gender = one_person['gender'] + two_person['gender']
                three_person = {
                    'length': self.past_len + self.future_len,
                    'trans': three_person_trans,
                    'poses': three_person_poses,
                    'betas': three_person_betas,
                    'gender': three_person_gender,
                    'global': 1,
                }

            else:
                index_one, frame_idx_one, bias_one, sr_one = self.idx2frame_one[idx - len(self.idx2frame_two)]
                one_person = copy.deepcopy(self.one_person_data[index_one])
                one_person_start_frame = np.random.choice(bias_one) + frame_idx_one
                one_person_end_frame = one_person_start_frame + (self.past_len + self.future_len) * sr_one
                idx_1 = np.random.choice(len(self.idx2frame_one))
                idx_2 = np.random.choice(len(self.idx2frame_one))
                index_1, frame_idx_1, bias_1, sr_1 = self.idx2frame_one[idx_1]
                one_person_1 = copy.deepcopy(self.one_person_data[index_1])
                one_person_start_frame_1 = np.random.choice(bias_1) + frame_idx_1
                one_person_end_frame_1 = one_person_start_frame_1 + (self.past_len + self.future_len) * sr_1
                index_2, frame_idx_2, bias_2, sr_2 = self.idx2frame_one[idx_2]
                one_person_2 = copy.deepcopy(self.one_person_data[index_2])
                one_person_start_frame_2 = np.random.choice(bias_2) + frame_idx_2
                one_person_end_frame_2 = one_person_start_frame_2 + (self.past_len + self.future_len) * sr_2

                one_person_trans = one_person['trans'][:, one_person_start_frame:one_person_end_frame]
                one_person_trans_1 = one_person_1['trans'][:, one_person_start_frame_1:one_person_end_frame_1]
                one_person_trans_2 = one_person_2['trans'][:, one_person_start_frame_2:one_person_end_frame_2]
                one_person_trans_range = torch.tensor([one_person_trans[:, :, 0].min() - one_person_trans[:, :, 0].max(), one_person_trans[:, :, 1].min() - one_person_trans[:, :, 1].max()]).unsqueeze(0)
                one_person_trans_range_1 = torch.tensor([one_person_trans_1[:, :, 0].min() - one_person_trans_1[:, :, 0].max(), one_person_trans_1[:, :, 1].min() - one_person_trans_1[:, :, 1].max()]).unsqueeze(0)
                one_person_trans_range_2 = torch.tensor([one_person_trans_2[:, :, 0].min() - one_person_trans_2[:, :, 0].max(), one_person_trans_2[:, :, 1].min() - one_person_trans_2[:, :, 1].max()]).unsqueeze(0)

                trans_range = -torch.cat([one_person_trans_range, one_person_trans_range_1, one_person_trans_range_2], dim=0)
                trans_range = trans_range.max(dim=0)[0] / 2

                rand_range = torch.rand(1)[0] * 3 + 1
                rand_value_0 = torch.rand(1)[0] * 2
                rand_value_1 = torch.rand(1)[0] * 4
                rand_value_2 = torch.rand(1)[0] * 3

                if rand_value_0 >= 1:
                    one_person_1['trans'][:, :, 0] -= one_person_trans_1[:, :, 0].max() - one_person_trans[:, :, 0].min() + rand_range
                    one_person_2['trans'][:, :, 0] -= one_person_trans_2[:, :, 0].min() - one_person_trans[:, :, 0].max() - rand_range
                if rand_value_0 <= 1.5:
                    if rand_value_2 > 1.5:
                        one_person_1['trans'][:, :, 1] -= one_person_trans_1[:, :, 1].max() - one_person_trans[:, :, 1].min() + rand_range
                        one_person_2['trans'][:, :, 1] -= one_person_trans_2[:, :, 1].min() - one_person_trans[:, :, 1].max() - rand_range
                    else:
                        one_person_2['trans'][:, :, 1] -= one_person_trans_2[:, :, 1].max() - one_person_trans[:, :, 1].min() + rand_range
                        one_person_1['trans'][:, :, 1] -= one_person_trans_1[:, :, 1].min() - one_person_trans[:, :, 1].max() - rand_range

                three_person_trans = torch.cat([one_person['trans'][:, one_person_start_frame:one_person_end_frame:sr_one], one_person_1['trans'][:, one_person_start_frame_1:one_person_end_frame_1:sr_1], one_person_2['trans'][:, one_person_start_frame_2:one_person_end_frame_2:sr_2]])
                three_person_poses = torch.cat([one_person['poses'][:, one_person_start_frame:one_person_end_frame:sr_one], one_person_1['poses'][:, one_person_start_frame_1:one_person_end_frame_1:sr_1], one_person_2['poses'][:, one_person_start_frame_2:one_person_end_frame_2:sr_2]])
                three_person_betas = torch.cat([one_person['betas'], one_person_1['betas'], one_person_2['betas']])
                three_person_gender = one_person['gender'] + one_person_1['gender'] + one_person_2['gender']
                three_person = {
                    'length': self.past_len + self.future_len,
                    'trans': three_person_trans,
                    'poses': three_person_poses,
                    'betas': three_person_betas,
                    'gender': three_person_gender,
                    'global': 0,
                }

        return three_person

    def __len__(self):
        return len(self.idx2frame_two) + len(self.idx2frame_one)
    

if __name__ == "__main__":
    from tqdm import tqdm
    from render.mesh_viz import visualize_body_multi
    from human_body_prior.body_model.body_model import BodyModel
    from os import path as osp
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("===========")
    sample_rate=4
    # dataset = Dataset(mode = 'train', past_len=10, future_len=25, sample_rate=sample_rate)
    dataset = Dataset(mode = 'train', past_len=10, future_len=25, sample_rate=sample_rate)
    # bm_fname = osp.join('./mocap', 'body_models/smplh/neutral/model.npz')
    # dmpl_fname = osp.join('./mocap', 'body_models/dmpls/neutral/model.npz')
    # bm = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(comp_device)
    # faces = c2c(bm.f)
    # # # print("===========")
    # outdir = "./results"
    # seq_save_path = outdir
    # os.makedirs(seq_save_path, exist_ok=True)
    # for k in tqdm(range(len(dataset))):
    #     records = dataset[k]
    #     # kinect_transform = KinectTransform(os.path.join(BEHAVE_PATH, records['seq_name']))
    #     # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    #     rend_video_path = os.path.join(seq_save_path, '{}.gif'.format(k))
    #     betas = records['betas'].unsqueeze(1).repeat(1, records['poses'].shape[1], 1)
    #     body_parms = {
    #         'root_orient': records['poses'].view(-1, records['poses'].shape[-1])[:, :3].float().to(comp_device), # controls the global root orientation
    #         'pose_body': records['poses'].view(-1, records['poses'].shape[-1])[:, 3:66].float().to(comp_device), # controls the body
    #         'pose_hand': records['poses'].view(-1, records['poses'].shape[-1])[:, 66:].float().to(comp_device), # controls the finger articulation
    #         'trans': records['trans'].view(-1, records['trans'].shape[-1]).float().to(comp_device), # controls the global body position
    #         'betas': betas.view(-1, records['betas'].shape[-1]).float().to(comp_device),# .to(comp_device), # controls the body shape. Body shape is static
    #     }
    #     body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'beta', 'pose_hand', 'trans', 'root_orient']})
        
    #     # print(np.argmin(jtr[:, :, 1], axis=1))
    #     verts = body_pose_hand.v.view(records['poses'].shape[0], records['poses'].shape[1], body_pose_hand.v.shape[1], body_pose_hand.v.shape[2]).cpu().numpy()

    #     m = visualize_body_multi(verts, faces, past_len=dataset.past_len, save_path=rend_video_path, sample_rate=1)
    bm_fname = osp.join('./mocap', 'body_models/smplh/neutral/model.npz')
    dmpl_fname = osp.join('./mocap', 'body_models/dmpls/neutral/model.npz')
    bm_neutral = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(comp_device)
    faces = c2c(bm_neutral.f)
    bm_fname = osp.join('./mocap', 'body_models/smplh/male/model.npz')
    dmpl_fname = osp.join('./mocap', 'body_models/dmpls/male/model.npz')
    bm_male = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(comp_device)
    bm_fname = osp.join('./mocap', 'body_models/smplh/female/model.npz')
    dmpl_fname = osp.join('./mocap', 'body_models/dmpls/female/model.npz')
    bm_female = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(comp_device)
    bms = {'neutral': bm_neutral, 'male': bm_male, 'female': bm_female}
    # # print("===========")
    outdir = "./results"
    seq_save_path = outdir
    os.makedirs(seq_save_path, exist_ok=True)
    for _ in tqdm(range(len(dataset))):
        k = torch.randint(0, len(dataset), ())
        records = dataset[k]
        # kinect_transform = KinectTransform(os.path.join(BEHAVE_PATH, records['seq_name']))
        # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
        rend_video_path = os.path.join(seq_save_path, '{}.gif'.format(k))
        betas = records['betas'].unsqueeze(1).repeat(1, records['poses'].shape[1], 1)
        verts = []
        for i in range(betas.shape[0]):
            bm = bms[records['gender'][i]]
            body_parms = {
                'root_orient': records['poses'][i][:, :3].float().to(comp_device), # controls the global root orientation
                'pose_body': records['poses'][i][:, 3:66].float().to(comp_device), # controls the body
                'pose_hand': records['poses'][i][:, 66:].float().to(comp_device), # controls the finger articulation
                'trans': records['trans'][i].float().to(comp_device), # controls the global body position
                'betas': betas[i].float().to(comp_device),# .to(comp_device), # controls the body shape. Body shape is static
            }
            body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'trans', 'root_orient']})
            verts.append(body_pose_hand.v.unsqueeze(0))
        # print(np.argmin(jtr[:, :, 1], axis=1))
        verts = torch.cat(verts, dim=0).cpu().numpy()

        m = visualize_body_multi(verts, faces, past_len=dataset.past_len, save_path=rend_video_path, sample_rate=1)
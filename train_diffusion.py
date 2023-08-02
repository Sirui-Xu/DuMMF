import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from DuMMF.Diffusion import *
from data_amass import *
import functools
from DuMMF.diffusion.resample import LossAwareSampler, UniformSampler
from DuMMF.diffusion.resample import create_named_schedule_sampler
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from render.mesh_viz import visualize_body_multi
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from utils.utils import point2point_signed, vertex_normals

class LitInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
        bm_fname = './mocap/body_models/smplh/neutral/model.npz'
        dmpl_fname = './mocap/body_models/dmpls/neutral/model.npz'
        bm_neutral = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(device)
        self.faces = c2c(bm_neutral.f)
        bm_fname = './mocap/body_models/smplh/male/model.npz'
        dmpl_fname = './mocap/body_models/dmpls/male/model.npz'
        bm_male = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(device)
        bm_fname = './mocap/body_models/smplh/female/model.npz'
        dmpl_fname = './mocap/body_models/dmpls/female/model.npz'
        bm_female = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname).to(device)
        self.body_model = {'neutral': bm_neutral, 'male': bm_male, 'female': bm_female}
        expr_dir = './mocap/body_models/V02_05'
        vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        self.vposer = vp.to(device)
        self.model, self.diffusion = create_model_and_diffusion(args)
        self.use_ddp = False
        self.ddp_model = self.model
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

    def on_train_start(self) -> None:
        #     backup trainer.py and model
        shutil.copy('./train_diffusion_v4.py', str(save_dir / 'train_diffusion.py'))
        shutil.copy('./DuMMF/Diffusion_v4.py', str(save_dir / 'Duffusion.py'))
        shutil.copy('./data_amass.py', str(save_dir / 'dataset.py'))
        shutil.copy('./DuMMF/diffusion/gaussian_diffusion.py', str(save_dir / 'diffusion.py'))
        return

    def penetration(self, vert, verts):
        face = torch.from_numpy(self.faces).unsqueeze(0).repeat(vert.shape[0], 1, 1).to(device)
        normals = vertex_normals(vert, face)
        o2h_signed, h2o_signed, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(vert, verts, x_normals=normals, return_vector=True)
        w = torch.zeros([o2h_signed.size(0), o2h_signed.size(1)]).to(device)
        w_dist = (o2h_signed < 0.01) * (o2h_signed >= 0)
        w_dist_neg = o2h_signed < 0
        w[w_dist] = 0 # small weight for far away vertices
        w[w_dist_neg] = 20 # large weight for penetration
        loss_dist_o = 1 * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed), w), dim=1) # 
        return loss_dist_o
    
    def pene_loss(self, body_rot_angle, body_trans, batch):
        T, B, nq, np, D = body_rot_angle.shape
        body_rot_angle = body_rot_angle.view(T, B * nq, np, -1)
        body_trans = body_trans.view(T * B * nq, np, -1)
        distance = (body_trans.unsqueeze(1) - body_trans.unsqueeze(2)).norm(dim=-1).min(dim=-1)[0]
        indices = distance.min(dim=1)[1] # T * B * nq
        betas = batch['betas'].unsqueeze(1).unsqueeze(0).repeat(T, 1, nq, 1, 1).view(T * B * nq * np, -1) # B, np, D
        body_para = torch.cat([body_rot_angle.view(T * B * nq * np, -1), body_trans.view(T * B * nq * np, -1)], dim=1)
        body_parms = {
            'root_orient': body_para[:, :3].float().to(device), # controls the global root orientation
            'pose_body': body_para[:, 3:66].float().to(device), # controls the body
            'pose_hand': body_para[:, 66:-3].float().to(device), # controls the finger articulation
            'trans': body_para[:, -3:].float().to(device), # controls the global body position
            'betas': betas.float().to(device),# .to(comp_device), # controls the body shape. Body shape is static
        }
        body_pose_hand = self.body_model['neutral'](**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'trans', 'root_orient']})
        verts = body_pose_hand.v.view(T * B * nq, np, -1, 3)
        indices = indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, np, verts.shape[2], verts.shape[3])
        vert = torch.gather(verts, 1, indices).view(T * B * nq * np, -1, 3)
        verts = verts.view(T * B * nq * np, -1, 3)
        loss = self.penetration(vert, verts)
        loss = loss.view(T, B, nq, np).mean(dim=[0, 2, 3])
        return loss
    
    def forward_backward(self, motion, cond, batch):
        t, weights = self.schedule_sampler.sample(self.args.batch_size, device)
        t = t.unsqueeze(1).unsqueeze(2).repeat(1, self.args.num_queries, self.args.num_persons).view(-1)
        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        cond['y']['global_level'] = annealing_factor
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            motion,
            t,
            model_kwargs=cond,
        )

        pred, gt = compute_losses()
        nq = self.args.num_queries
        np = self.args.num_persons
        pred = pred.squeeze(1).permute(2, 0, 1).contiguous()
        gt = gt.squeeze(1).permute(2, 0, 1).contiguous()
        T, _, nJ = pred[:, :, :-3].shape
        pred = pred.view(T, -1, nq, np, nJ + 3)
        gt = gt.view(T, -1, nq, np, nJ + 3)
        nJ = nJ // 6
        body_rot = pred[:, :, :, :, :-3]
        body_rot_gt = gt[:, :, :, :, :-3]
        body_trans = pred[:, :, :, :, -3:]
        body_trans_gt = gt[:, :, :, :, -3:]
        body_rot_angle = matrix_to_axis_angle(rotation_6d_to_matrix(body_rot.view(T, -1, nq, np, nJ, 6))).view(T, -1, nq, np, nJ * 3)
        
        loss_body_rot_past_global = torch.nn.MSELoss(reduction='none')(body_rot[:self.args.past_len], body_rot_gt[:self.args.past_len]).mean(dim=[0, 2, 3, 4])
        loss_body_nonrot_past_global = torch.nn.MSELoss(reduction='none')(body_trans[:self.args.past_len], body_trans_gt[:self.args.past_len]).mean(dim=[0, 2, 3, 4])

        loss_body_rot_v_past_global = torch.nn.MSELoss(reduction='none')(body_rot[1:self.args.past_len+1]-body_rot[:self.args.past_len], body_rot_gt[1:self.args.past_len+1]-body_rot_gt[1:self.args.past_len+1]).mean(dim=[0, 2, 3, 4]) +\
                                      torch.nn.MSELoss(reduction='none')(body_rot[1:self.args.past_len]-body_rot[:self.args.past_len-1], body_rot[2:self.args.past_len+1]-body_rot[1:self.args.past_len]).mean(dim=[0, 2, 3, 4])
        loss_body_nonrot_v_past_global = torch.nn.MSELoss(reduction='none')(body_trans[1:self.args.past_len+1]-body_trans[:self.args.past_len], body_trans_gt[1:self.args.past_len+1]-body_trans_gt[1:self.args.past_len+1]).mean(dim=[0, 2, 3, 4]) +\
                                         torch.nn.MSELoss(reduction='none')(body_trans[1:self.args.past_len]-body_trans[:self.args.past_len-1], body_trans[2:self.args.past_len+1]-body_trans[1:self.args.past_len]).mean(dim=[0, 2, 3, 4])

        loss_body_rot_future_global = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:], body_rot_gt[self.args.past_len:]).mean(dim=[0, 3, 4])
        loss_body_nonrot_future_global = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:], body_trans_gt[self.args.past_len:]).mean(dim=[0, 3, 4])

        loss_body_rot_v_future_global = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:]-body_rot[self.args.past_len-1:-1], body_rot_gt[self.args.past_len:]-body_rot_gt[self.args.past_len:]).mean(dim=[0, 2, 3, 4]) +\
                                        torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len-1:-2]-body_rot[self.args.past_len:-1], body_rot[self.args.past_len:-1]-body_rot[self.args.past_len+1:]).mean(dim=[0, 2, 3, 4])
        loss_body_nonrot_v_future_global = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:]-body_trans[self.args.past_len-1:-1], body_trans_gt[self.args.past_len:]-body_trans_gt[self.args.past_len:]).mean(dim=[0, 2, 3, 4]) +\
                                           torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len-1:-2]-body_trans[self.args.past_len:-1], body_trans[self.args.past_len:-1]-body_trans[self.args.past_len+1:]).mean(dim=[0, 2, 3, 4])

        # idx_q = torch.randint(0, self.args.num_queries, ())
        # if idx_q % 2 == 0:
        #     loss_pene = self.pene_loss(body_rot_angle[self.args.past_len:, :, idx_q:idx_q+1, :, :].contiguous(), body_trans[self.args.past_len:, :, idx_q:idx_q+1, :, :].contiguous(), batch)
        # else:
        loss_pene = torch.zeros_like(loss_body_rot_past_global).to(device)

        cond['y']['global_level'] = 0
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            motion,
            t,
            model_kwargs=cond,
        )

        pred_local, gt_local = compute_losses()
        pred_local = pred_local.squeeze(1).permute(2, 0, 1).contiguous()
        gt_local = gt_local.squeeze(1).permute(2, 0, 1).contiguous()
        T, _, nJ = pred_local[:, :, :-3].shape
        pred_local = pred_local.view(T, -1, nq, np, nJ + 3)
        gt_local = gt_local.view(T, -1, nq, np, nJ + 3)
        nJ = nJ // 6
        body_rot = pred_local[:, :, :, :, :-3]
        body_rot_gt = gt_local[:, :, :, :, :-3]
        body_trans = pred_local[:, :, :, :, -3:]
        body_trans_gt = gt_local[:, :, :, :, -3:]
        body_rot_angle_local = matrix_to_axis_angle(rotation_6d_to_matrix(body_rot.view(T, -1, nq, np, nJ, 6))).view(T, -1, nq, np, nJ * 3)

        loss_body_rot_past_local = torch.nn.MSELoss(reduction='none')(body_rot[:self.args.past_len], body_rot_gt[:self.args.past_len]).mean(dim=[0, 4]).mean(dim=1).mean(dim=1)
        loss_body_nonrot_past_local = torch.nn.MSELoss(reduction='none')(body_trans[:self.args.past_len], body_trans_gt[:self.args.past_len]).mean(dim=[0, 4]).mean(dim=1).mean(dim=1)

        loss_body_rot_v_past_local = torch.nn.MSELoss(reduction='none')(body_rot[1:self.args.past_len+1]-body_rot[:self.args.past_len], body_rot_gt[1:self.args.past_len+1]-body_rot_gt[1:self.args.past_len+1]).mean(dim=[0, 2, 3, 4]) +\
                                     torch.nn.MSELoss(reduction='none')(body_rot[1:self.args.past_len]-body_rot[:self.args.past_len-1], body_rot[2:self.args.past_len+1]-body_rot[1:self.args.past_len]).mean(dim=[0, 2, 3, 4])
                                     
        loss_body_nonrot_v_past_local = torch.nn.MSELoss(reduction='none')(body_trans[1:self.args.past_len+1]-body_trans[:self.args.past_len], body_trans_gt[1:self.args.past_len+1]-body_trans_gt[1:self.args.past_len+1]).mean(dim=[0, 2, 3, 4]) +\
                                        torch.nn.MSELoss(reduction='none')(body_trans[1:self.args.past_len]-body_trans[:self.args.past_len-1], body_trans[2:self.args.past_len+1]-body_trans[1:self.args.past_len]).mean(dim=[0, 2, 3, 4])

        loss_body_rot_future_local = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:], body_rot_gt[self.args.past_len:]).mean(dim=[0, 4])
        loss_body_nonrot_future_local = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:], body_trans_gt[self.args.past_len:]).mean(dim=[0, 4])

        loss_body_rot_v_future_local = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:]-body_rot[self.args.past_len-1:-1], body_rot_gt[self.args.past_len:]-body_rot_gt[self.args.past_len:]).mean(dim=[0, 2, 3, 4]) +\
                                       torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len-1:-2]-body_rot[self.args.past_len:-1], body_rot[self.args.past_len:-1]-body_rot[self.args.past_len+1:]).mean(dim=[0, 2, 3, 4])
        loss_body_nonrot_v_future_local = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:]-body_trans[self.args.past_len-1:-1], body_trans_gt[self.args.past_len:]-body_trans_gt[self.args.past_len:]).mean(dim=[0, 2, 3, 4]) +\
                                          torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len-1:-2]-body_trans[self.args.past_len:-1], body_trans[self.args.past_len:-1]-body_trans[self.args.past_len+1:]).mean(dim=[0, 2, 3, 4])

        if self.args.num_queries == 1:
            loss_div_rot = torch.zeros_like(loss_body_rot_past_local).to(device)
            loss_div_trans = torch.zeros_like(loss_body_rot_past_local).to(device)
        else:
            mask = torch.tril(torch.ones([nq, nq], device=device)) == 0
            body_rot_view = body_rot[self.args.past_len:].permute(1, 3, 2, 0, 4).contiguous().view(-1, nq, self.args.future_len * nJ * 6)
            body_trans_view = body_trans[self.args.past_len:].permute(1, 3, 2, 0, 4).contiguous().view(-1, nq, self.args.future_len * 3)
            pdist_rot = torch.cdist(body_rot_view, body_rot_view, p=1)[:, mask]
            pdist_trans = torch.cdist(body_trans_view, body_trans_view, p=1)[:, mask]
            loss_div_rot = (-pdist_rot / 20000).exp().mean(dim=-1).view(-1, np).mean(dim=-1)
            loss_div_trans = (-pdist_trans / 800).exp().mean(dim=-1).view(-1, np).mean(dim=-1)
        
        pred_view = body_rot_angle[self.args.past_len:, :, :, :, 3:66].contiguous().view(-1, 63)
        amass_body_poZ = self.vposer.encode(pred_view).mean
        loss_prior_global = amass_body_poZ.pow(2).sum(dim=1).view(self.args.future_len, -1, nq, np).mean(dim=[0, 2, 3])

        pred_view = body_rot_angle_local[self.args.past_len:, :, :, :, 3:66].contiguous().view(-1, 63)
        amass_body_poZ = self.vposer.encode(pred_view).mean
        loss_prior_local = amass_body_poZ.pow(2).sum(dim=1).view(self.args.future_len, -1, nq, np).mean(dim=[0, 2, 3])

        loss_dict = dict()
        weighted_loss_dict = dict()

        body_rot_future=loss_body_rot_future_global * self.args.weight_smplx_rot
        body_nonrot_future=loss_body_nonrot_future_global * self.args.weight_smplx_nonrot
        
        global_future = body_rot_future + body_nonrot_future

        body_rot_future=loss_body_rot_future_local * self.args.weight_smplx_rot
        body_nonrot_future=loss_body_nonrot_future_local * self.args.weight_smplx_nonrot
        
        local_future = body_rot_future + body_nonrot_future

        if torch.rand(()) < 0.9 * annealing_factor:
            future = batch['global'] * global_future.min(dim=1)[0] + (1 - batch['global']) * local_future.min(dim=1)[0].mean(dim=-1) * annealing_factor
        else:
            idx_q = torch.randint(0, self.args.num_queries, ())
            future = batch['global'] * global_future[:, idx_q] + (1 - batch['global']) * local_future[:, idx_q].mean(dim=1) * annealing_factor

        loss_dict.update(dict(
                        body_rot_past = batch['global'] * loss_body_rot_past_global + (1 - batch['global']) * loss_body_rot_past_local * annealing_factor,
                        body_nonrot_past = batch['global'] * loss_body_nonrot_past_global + (1 - batch['global']) * loss_body_nonrot_past_local * annealing_factor,
                        body_rot_v_past = batch['global'] * loss_body_rot_v_past_global + (1 - batch['global']) * loss_body_rot_v_past_local * annealing_factor,
                        body_nonrot_v_past = batch['global'] * loss_body_nonrot_v_past_global + (1 - batch['global']) * loss_body_nonrot_v_past_local * annealing_factor,
                        body_rot_v_future = batch['global'] * loss_body_rot_v_future_global + (1 - batch['global']) * loss_body_rot_v_future_local * annealing_factor,
                        body_nonrot_v_future = batch['global'] * loss_body_nonrot_v_future_global + (1 - batch['global']) * loss_body_nonrot_v_future_local * annealing_factor,
                        future = future,
                        div_rot=loss_div_rot,
                        div_trans=loss_div_trans,
                        prior_global=loss_prior_global,
                        prior_local=loss_prior_local * annealing_factor,
                        penetration=loss_pene,
                        ))
        weighted_loss_dict.update(dict(
                                body_rot_past=loss_dict['body_rot_past'] * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_past=loss_dict['body_nonrot_past'] * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body_rot_v_past=loss_dict['body_rot_v_past'] * self.args.weight_v * self.args.weight_smplx_rot * self.args.weight_past,
                                body_nonrot_v_past=loss_dict['body_nonrot_v_past'] * self.args.weight_v * self.args.weight_smplx_nonrot * self.args.weight_past,
                                body_rot_v_future=loss_dict['body_rot_v_future'] * self.args.weight_v * self.args.weight_smplx_rot,
                                body_nonrot_v_future=loss_dict['body_nonrot_v_future'] * self.args.weight_v * self.args.weight_smplx_nonrot,
                                future = future,
                                div_rot=loss_div_rot * self.args.weight_div * max(annealing_factor ** 2, 0),
                                div_trans=loss_div_trans * self.args.weight_div * max(annealing_factor ** 2, 0),
                                prior_global=loss_prior_global * self.args.weight_prior,
                                prior_local=loss_prior_local * self.args.weight_prior,
                                penetration=loss_pene * self.args.weight_penetration * max(annealing_factor ** 2, 0),
                                ))
        
        loss = sum(list(weighted_loss_dict.values()))

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss.detach()
            )

        loss = (loss * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, weighted_loss_dict, loss
        )
        return loss

    def log_loss_dict(self, diffusion, ts, losses, loss):
        self.log('train_loss', loss, prog_bar=False)
        for key, values in losses.items():
            self.log(key, values.mean().item(), prog_bar=True)
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.model.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 # 'lr_scheduler': {
                 #    'scheduler': lr_scheduler,
                 #    'reduce_on_plateau': True,
                 #    # val_checkpoint_on is val_loss passed in as checkpoint_on
                 #    'monitor': 'joint'
                 #    }
                 })

    def calc_val_loss(self, pred, gt, batch):
        nq = self.args.num_queries
        np = self.args.num_persons

        T, _, nJ = pred[:, :, :-3].shape
        pred = pred.view(T, -1, nq, np, nJ + 3)
        gt = gt.view(T, -1, nq, np, nJ + 3)
        nJ = nJ // 6
        body_rot = pred[:, :, :, :, :-3]
        body_rot_gt = gt[:, :, :, :, :-3]
        body_trans = pred[:, :, :, :, -3:]
        body_trans_gt = gt[:, :, :, :, -3:]

        loss_body_rot_future_global = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:], body_rot_gt[self.args.past_len:]).mean(dim=[0, 3, 4]).min(dim=1)[0].mean()
        loss_body_nonrot_future_global = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:], body_trans_gt[self.args.past_len:]).mean(dim=[0, 3, 4]).min(dim=1)[0].mean()

        loss_body_rot_v_future_global = torch.nn.MSELoss(reduction='none')(body_rot[self.args.past_len:]-body_rot[self.args.past_len-1:-1], body_rot_gt[self.args.past_len:]-body_rot_gt[self.args.past_len-1:-1]).mean(dim=[0, 3, 4]).min(dim=1)[0].mean()
        loss_body_nonrot_v_future_global = torch.nn.MSELoss(reduction='none')(body_trans[self.args.past_len:]-body_trans[self.args.past_len-1:-1], body_trans_gt[self.args.past_len:]-body_trans_gt[self.args.past_len-1:-1]).mean(dim=[0, 3, 4]).min(dim=1)[0].mean()
        
        body_rot_angle = matrix_to_axis_angle(rotation_6d_to_matrix(body_rot.view(T, -1, nq, np, nJ, 6))).view(T, -1, nq, np, nJ * 3)
        body_rot_gt_angle = matrix_to_axis_angle(rotation_6d_to_matrix(body_rot_gt.view(T, -1, nq, np, nJ, 6))).view(T, -1, nq, np, nJ * 3)
        pred_ = torch.cat([body_rot_angle, body_trans], dim=4)
        gt_ = torch.cat([body_rot_gt_angle, body_trans_gt], dim=4)

        pred_view = body_rot_angle[self.args.past_len:, :, :, :, 3:66].contiguous().view(-1, 63)
        amass_body_poZ = self.vposer.encode(pred_view).mean
        loss_prior_global = amass_body_poZ.pow(2).sum(dim=1).view(self.args.future_len, -1, nq, np).mean(dim=[0, 2, 3]).mean()

        if self.args.num_queries == 1:
            loss_div_rot = torch.zeros_like(loss_body_rot_future_global).to(device)
            loss_div_trans = torch.zeros_like(loss_body_rot_future_global).to(device)
        else:
            mask = torch.tril(torch.ones([nq, nq], device=device)) == 0
            body_rot_view = body_rot[self.args.past_len:].permute(1, 3, 2, 0, 4).contiguous().view(-1, nq, self.args.future_len * nJ * 6)
            body_trans_view = body_trans[self.args.past_len:].permute(1, 3, 2, 0, 4).contiguous().view(-1, nq, self.args.future_len * 3)
            pdist_rot = torch.cdist(body_rot_view, body_rot_view, p=1)[:, mask]
            pdist_trans = torch.cdist(body_trans_view, body_trans_view, p=1)[:, mask]
            loss_div_rot = pdist_rot.mean()
            loss_div_trans = pdist_trans.mean()
        
        # idx_q = torch.randint(0, self.args.num_queries, ())
        # loss_pene = self.pene_loss(body_rot_angle[self.args.past_len:, :, idx_q:idx_q+1, :, :].contiguous(), body_trans[self.args.past_len:, :, idx_q:idx_q+1, :, :].contiguous(), batch).mean()

        loss_dict = dict()
        weighted_loss_dict = dict()
        loss_dict.update(dict(
                        body_rot_future = loss_body_rot_future_global,
                        body_nonrot_future = loss_body_nonrot_future_global,
                        body_rot_v_future = loss_body_rot_v_future_global,
                        body_nonrot_v_future = loss_body_nonrot_v_future_global,
                        div_rot=loss_div_rot,
                        div_trans=loss_div_trans,
                        prior_global=loss_prior_global,
                        # penetration=loss_pene,
                        ))
        weighted_loss_dict.update(dict(
                                body_rot_future=loss_dict['body_rot_future'] * self.args.weight_smplx_rot,
                                body_nonrot_future=loss_dict['body_nonrot_future'] * self.args.weight_smplx_nonrot,
                                body_rot_v_future=loss_dict['body_rot_v_future'] * self.args.weight_v * self.args.weight_smplx_rot,
                                body_nonrot_v_future=loss_dict['body_nonrot_v_future'] * self.args.weight_v * self.args.weight_smplx_nonrot,
                                div_rot=loss_div_rot * self.args.weight_div,
                                div_trans=loss_div_trans * self.args.weight_div,
                                prior_global=loss_prior_global * self.args.weight_prior,
                                # penetration=loss_pene * self.args.weight_penetration,
                                ))
        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict, pred_, gt_

    def _common_step(self, batch, batch_idx, mode):
        embedding, gt = self.model._get_embeddings(batch)
        # [t, b, n] -> [bs, njoints, nfeats, nframes]
        gt = gt.permute(1, 2, 0).unsqueeze(1).contiguous()
        model_kwargs = {'y': {'cond': embedding}}
        model_kwargs['y']['inpainted_motion'] = gt
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt, dtype=torch.bool,
                                                                device=device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, self.args.past_len:] = False  # do inpainting in those frames
        if mode == 'train':
            loss = self.forward_backward(gt, model_kwargs, batch)
            return loss
        elif mode == 'valid' or mode == 'test':
            model_kwargs['y']['global_level'] = 1
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(self.model, gt.shape, clip_denoised=False, model_kwargs=model_kwargs)
            pred = sample.squeeze(1).permute(2, 0, 1).contiguous()
            gt = gt.squeeze(1).permute(2, 0, 1).contiguous()
            loss, loss_dict, weighted_loss_dict, pred, gt = self.calc_val_loss(pred, gt, batch)
            
            render_interval = 100
            if (batch_idx % render_interval == 0) and (((self.current_epoch % self.args.render_epoch) == self.args.render_epoch - 1) or self.args.debug):
                nq = self.args.num_queries
                for i in range(nq+1):
                    if i == nq:
                        self.visualize(gt[:, 0, 0, :, :], batch, batch_idx, 'gt', i)
                    else:
                        self.visualize(pred[:, 0, i, :, :], batch, batch_idx, 'pred', i)
            return loss, loss_dict, weighted_loss_dict

    def visualize(self, pred, batch, batch_idx, mode, idx):
        with torch.no_grad():
            pred = pred.detach().clone()
            betas = batch['betas'][0].unsqueeze(1).repeat(1, pred.shape[0], 1) # np T D
            trans = batch['trans'][0, :, 0:1, :] # MxTx3
            gender = batch['gender'] # np 
            # visualize
            export_file = Path.joinpath(save_dir, 'render')
            export_file.mkdir(exist_ok=True, parents=True)
            # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
            rend_video_path = os.path.join(export_file, '{}_{}_{}_{}'.format(mode, self.current_epoch, batch_idx, idx))
            
            verts = []
            for i in range(betas.shape[0]):
                bm = self.body_model[gender[i][0]]
                body_parms = {
                    'root_orient': pred[:, i, :3].float().to(device), # controls the global root orientation
                    'pose_body': pred[:, i, 3:66].float().to(device), # controls the body
                    'pose_hand': pred[:, i, 66:-3].float().to(device), # controls the finger articulation
                    'trans': pred[:, i, -3:].float().to(device) + trans[i].float().to(device), # controls the global body position
                    'betas': betas[i].float().to(device),# .to(comp_device), # controls the body shape. Body shape is static
                }
                body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'trans', 'root_orient']})
                verts.append(body_pose_hand.v.unsqueeze(0))
            # print(np.argmin(jtr[:, :, 1], axis=1))
            verts = torch.cat(verts, dim=0).cpu().numpy()
    
            m = visualize_body_multi(verts, self.faces, past_len=self.args.past_len, save_path=rend_video_path, sample_rate=1)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'test')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--sample_rate", type=int, default=4)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--latent_usage", type=str, default='memory')

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_penetration", type=float, default=0.01)  #10

    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.2)
    parser.add_argument("--weight_past", type=float, default=1)
    parser.add_argument("--weight_jtr", type=float, default=0.1)
    parser.add_argument("--weight_jtr_v", type=float, default=500)
    parser.add_argument("--weight_v", type=float, default=0.2)
    parser.add_argument("--weight_div", type=float, default=0.0001)
    parser.add_argument("--weight_prior", type=float, default=0)

    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=1)

    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--num_persons", type=int, default=3)
    # dataset
    parser.add_argument("--past_len", type=int, default=10)
    parser.add_argument("--future_len", type=int, default=25)

    # train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--second_stage", type=int, default=100,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_epoch", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)

    # diffusion
    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--cond_mask_prob", default=0, type=float,
                        help="The probability of masking the condition during training."
                             " For classifier-free guidance learning.")
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # rendering and results
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    train_dataset = Dataset(mode = 'train', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)
    test_dataset = Dataset(mode = 'test', past_len=args.past_len, future_len=args.future_len, sample_rate=args.sample_rate)

    args.smpl_dim = train_dataset.pose_dim * 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)  #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitInteraction(args)

    if args.mode == "train":
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/interaction'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                gradient_clip_val=0.01,
                                                check_val_every_n_epoch=250,
                                                )
        trainer.fit(model, train_loader, val_loader)

    elif args.mode == "test" and args.resume_checkpoint is not None:
        # callback
        tb_logger = pl_loggers.TensorBoardLogger(str(results_folder + '/sample'), name=args.expr_name)
        save_dir = Path(tb_logger.log_dir)  # for this version
        print(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                        monitor="val_loss",
                                                        save_weights_only=True, save_last=True)
        print(checkpoint_callback.dirpath)
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                        mode="min")
        profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

        # trainer
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                profiler=profiler,
                                                # progress_bar_refresh_rate=1,
                                                callbacks=[checkpoint_callback, early_stop_callback],
                                                )
        trainer.test(model, val_loader)



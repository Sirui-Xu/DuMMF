import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from DuMMF.SubLayers import TransformerEncoderLayer, TransformerEncoderLayerQaN
from DuMMF.Layers import PositionalEncoding, TimestepEmbedder, TransformerEncoder

class MDM(nn.Module):
    def __init__(self, args):
        super(MDM, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.bodyEmbedding = nn.Linear(args.smpl_dim+3, num_channels)
        self.globalEmbedding = nn.Linear(9, num_channels)
        self.PositionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        self.embedTimeStep = TimestepEmbedder(num_channels, self.PositionalEmbedding)
        seqTransDecoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransDecoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            num_queries=self.args.num_queries,
                                                            num_persons=self.args.num_persons,
                                                            batch_first=False)
        seqTransDecoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransDecoderLayer = nn.ModuleList([seqTransDecoderLayer1, seqTransDecoderLayer2, seqTransDecoderLayer3, seqTransDecoderLayer4,
                                              seqTransDecoderLayer5, seqTransDecoderLayer6, seqTransDecoderLayer7, seqTransDecoderLayer8])
        self.decoder = TransformerEncoder(seqTransDecoderLayer)

        self.finalLinear = nn.Linear(num_channels, args.smpl_dim+3)

    def mask_cond(self, cond, force_mask=False):
        t, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.args.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.args.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def _get_embeddings(self, data, device=None):
        if device:
            body_pose = data['poses'].float().to(device) # BxMxTxD
            body_trans = data['trans'].float().to(device)  # BxMxTx3

        else:
            body_pose = data['poses'].float() # BxMxTxD
            body_trans = data['trans'].float() # BxMxTx3

        B, M, T, _ = body_pose.shape
        body_pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose.view(B, M, T, -1, 3))).view(B, M, T, -1)
        body_trans_start = body_trans[:, :, 0:1, :]
        body_pose_start = body_pose[:, :, 0:1, :6]
        body_trans = body_trans - body_trans_start
        body_start = torch.cat([body_trans_start, body_pose_start], dim=3)
        embedding = self.globalEmbedding(body_start).unsqueeze(3).repeat(1, 1, self.args.past_len+self.args.future_len, self.args.num_queries, 1).permute(2, 0, 3, 1, 4).contiguous().view(self.args.past_len+self.args.future_len, B*self.args.num_queries*M, -1)
        # print('model convert', obj_angles[0])
        gt = torch.cat([body_pose, body_trans], dim=3)
        # B N M T D -> T BNM D
        gt = gt.unsqueeze(1).repeat(1, self.args.num_queries, 1, 1, 1).permute(3, 0, 1, 2, 4).view(T, B*self.args.num_queries*M, -1)
        return embedding, gt


    def _decode(self, x, time_embedding, y=None, global_level=True):
        body = self.bodyEmbedding(x)
        decoder_input = body + time_embedding
        decoder_input = self.PositionalEmbedding(decoder_input)
        decoder_output = self.decoder(decoder_input, memory=y, global_level=global_level)
        body = self.finalLinear(decoder_output)
        return body

    def forward(self, x, timesteps, y=None):
        # print(timesteps)
        time_embedding = self.embedTimeStep(timesteps)
        x = x.squeeze(1).permute(2, 0, 1).contiguous()
        cond = y['cond']
        x_0 = self._decode(x, time_embedding, cond, y['global_level'])
        # [T B N] -> [bs, njoints, nfeats, nframes]
        x_0 = x_0.permute(1, 2, 0).unsqueeze(1).contiguous()
        return x_0

from DuMMF.diffusion import gaussian_diffusion as gd
from DuMMF.diffusion.respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.weight_v,
    )

def create_model_and_diffusion(args):
    model = MDM(args)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion
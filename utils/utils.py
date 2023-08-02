import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import smplx
import chamfer_distance as chd

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)
def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
    
def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y
    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y,x_normals=x_normals,y_normals=y_normals)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y
    
SMPLH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

SMPLH_VERTEX_INDEX = {
    'nose':		    332,
    'reye':		    6260,
    'leye':		    2800,
    'rear':		    4071,
    'lear':		    583,
    'rthumb':		6191,
    'rindex':		5782,
    'rmiddle':		5905,
    'rring':		6016,
    'rpinky':		6133,
    'lthumb':		2746,
    'lindex':		2319,
    'lmiddle':		2445,
    'lring':		2556,
    'lpinky':		2673,
    'LBigToe':		3216,
    'LSmallToe':	3226,
    'LHeel':		3387,
    'RBigToe':		6617,
    'RSmallToe':    6624,
    'RHeel':		6787
}

class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=SMPLH_VERTEX_INDEX,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        # face_keyp_idxs = np.array([
        #     vertex_ids['nose'],
        #     vertex_ids['reye'],
        #     vertex_ids['leye'],
        #     vertex_ids['rear'],
        #     vertex_ids['lear']], dtype=np.int64)

        # extra_joints_idxs = np.concatenate([extra_joints_idxs,
        #                                     face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(torch.long)) #The '.to(torch.long)'.
                                                                                            # added to make the trace work in c++,
                                                                                            # otherwise you get a runtime error in c++:
                                                                                            # 'index_select(): Expected dtype int32 or int64 for index'
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints

marker_dict = {'C7': 3470, 'CLAV': 3171, 'LANK': 3327, 'LASI': 857, 'LBAK': 1812, 
               'LBCEP': 628, 'LBHD': 182, 'LBUM': 3116, 'LBUST': 3040, 'LCHEECK': 239,
               'LELB': 1666, 'LELBIN': 1725, 'LFHD': 0, 'LFIN': 2174, 'LFRM': 1568,
               'LFTHIIN': 1368, 'LHEE': 3387, 'LIWR': 2112, 'LKNE': 1053, 'LKNI': 1058,
               'LMT1': 3336, 'LMT5': 3346, 'LNWST': 1323, 'LOWR': 2108, 'LPSI': 3122,
               'LRSTBEEF': 3314, 'LSCAP': 1252, 'LSHN': 1082, 'LSHO': 1861, 'LTHI': 1454,
               'LTHILO': 850, 'LTHMB': 2224, 'LTOE': 3233, 'MBLLY': 1769, 'RANK': 6728,
               'RASI': 4343, 'RBAK': 5273, 'RBCEP': 4116, 'RBHD': 3694, 'RBSH': 6399,
               'RBUM': 6540, 'RBUST': 6488, 'RCHEECK': 3749, 'RELB': 5135, 'RELBIN': 5194,
               'RFHD': 3512, 'RFIN': 5635, 'RFRM2': 5210, 'RFTHI': 4360, 'RFTHIIN': 4841,
               'RHEE': 6786, 'RIWR': 5573, 'RKNE': 4538, 'RKNI': 4544, 'RMT1': 6736,
               'RMT5': 6747, 'RNWST': 4804, 'ROWR': 5568, 'RPSI': 6544, 'RRSTBEEF': 6682,
               'RSHO': 5322, 'RTHI': 4927, 'RTHMB': 5686, 'RTIB': 4598, 'RTOE': 6633,
               'STRN': 3506, 'T8': 3508}

markerset_ssm67_smplh = [3470, 3171, 3327, 857, 1812, 628, 182, 3116, 3040, 239,
                         1666, 1725, 0, 2174, 1568, 1368, 3387, 2112, 1053, 1058,
                         3336, 3346, 1323, 2108, 3122, 3314, 1252, 1082, 1861, 1454,
                         850, 2224, 3233, 1769, 6728, 4343, 5273, 4116, 3694, 6399,
                         6540, 6488, 3749, 5135, 5194, 3512, 5635, 5210, 4360, 4841,
                         6786, 5573, 4538, 4544, 6736, 6747, 4804, 5568, 6544, 6682,
                         5322, 4927, 5686, 4598, 6633, 3506, 3508]

markerset_wfinger = [3470, 3171, 3327, 857, 1812, 628, 182, 3116, 3040, 239,
                         1666, 1725, 0, 2174, 1568, 1368, 3387, 2112, 1053, 1058,
                         3336, 3346, 1323, 2108, 3122, 3314, 1252, 1082, 1861, 1454,
                         850, 2224, 3233, 1769, 6728, 4343, 5273, 4116, 3694, 6399,
                         6540, 6488, 3749, 5135, 5194, 3512, 5635, 5210, 4360, 4841,
                         6786, 5573, 4538, 4544, 6736, 6747, 4804, 5568, 6544, 6682,
                         5322, 4927, 5686, 4598, 6633, 3506, 3508,
                         6191, 5782, 5905, 6016, 6133, 2746, 2319, 2445, 2556, 2673]

marker2bodypart = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55],
    "left_finger_ids": [72, 73, 74, 75, 76],
    "right_finger_ids": [67, 68, 69, 70, 71],
}

marker2bodypart67 = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55],
}

bodypart2color = {
    "head_ids": 'cyan',
    "mid_body_ids": 'blue',
    "left_hand_ids": 'red',
    "right_hand_ids": 'green',
    "left_foot_ids": 'grey',
    "right_foot_ids": 'black',
    "left_toe_ids": 'yellow',
    "right_toe_ids": 'magenta',
    "left_finger_ids": 'red',
    "right_finger_ids": 'green',
    "special": 'light_grey'
}


colors = {
        "cyan": [0, 255, 255, 1],
        "yellow": [255, 255, 0, 1],
        "blue": [162, 26, 15, 1],
        "grey": [77, 77, 77, 1],
        "grey_transparent": [77, 77, 77, 0.1],
        "ultra_bright_grey": [200, 200, 200, 1],
        "black": [0, 0, 0, 1],
        "white": [255, 255, 255, 1],
        "transparent": [255, 255, 255, 0],
        "magenta": [197, 27, 125, 1],
        'pink': [197, 140, 133, 1],
        'pink_transparent': [197, 140, 133, 0.1],
        "light_grey": [217, 217, 217, 255],
        "light_grey_transparent": [217, 217, 217, 0.1],
        'red': [26, 15, 162, 1],
        'green': [26, 162, 15, 1],
        'yellow_pale': [226, 215, 132, 1],
        'yellow_pale_transparent': [226, 215, 132, 0.1],
        "color0": [200, 200, 200, 1],
        "color1": [226, 152, 178, 1],
        "color2": [152, 226, 178, 1],
        "color3": [152, 178, 226, 1],
        "color4": [226, 178, 152, 1],
        "color5": [178, 226, 152, 1],
        "color6": [178, 152, 226, 1],
        "color7": [226, 120, 200, 1],
        "color8": [120, 226, 200, 1],
        "color9": [120, 200, 226, 1],
        "color10": [226, 200, 120, 1],
        "color11": [200, 226, 120, 1],
        "color12": [200, 120, 226, 1],
        "color13": [226, 120, 245, 1],
        "color14": [226, 245, 178, 1],
        "color15": [245, 226, 178, 1],
        "color16": [245, 178, 226, 1],
        "color17": [226, 178, 245, 1],
        "color18": [178, 226, 245, 1],

        }


colors_rgb = {
        "blue": [0, 0, 255/255],
        "cyan": [0, 128/255, 255/255],
        "green": [0, 255/255, 0],
        "yellow": [255/255, 255/255, 0],
        "red": [255/255, 0, 0],
        "grey": [77/255, 77/255, 77/255],
        "black": [0, 0, 0],
        "white": [255/255, 255/255, 255/255],
        "magenta": [197/255, 27, 125/255],
        'pink': [197/255, 140/255, 133/255],
        "light_grey": [217/255, 217/255, 217/255],
        }

mmm_joints = ["root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
              "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"]

smplh_joints = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee",
                "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
                "left_foot", "right_foot", "neck", "left_collar", "right_collar",
                "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3",
                "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
                "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
                "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3",
                "right_middle1", "right_middle2", "right_middle3", "right_pinky1",
                "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3",
                "right_thumb1", "right_thumb2", "right_thumb3", "nose", "right_eye", "left_eye",
                "right_ear", "left_ear", "left_big_toe", "left_small_toe", "left_heel",
                "right_big_toe", "right_small_toe", "right_heel", "left_thumb", "left_index",
                "left_middle", "left_ring", "left_pinky", "right_thumb", "right_index",
                "right_middle", "right_ring", "right_pinky"]

mmm2smplh_correspondence = {"root": "pelvis", "BP": "spine1", "BT": "spine3", "BLN": "neck", "BUN": "head",
                            "LS": "left_shoulder", "LE": "left_elbow", "LW": "left_wrist",
                            "RS": "right_shoulder", "RE": "right_elbow", "RW": "right_wrist",
                            "LH": "left_hip", "LK": "left_knee", "LA": "left_ankle", "LMrot": "left_heel",
                            "LF": "left_foot",
                            "RH": "right_hip", "RK": "right_knee", "RA": "right_ankle", "RMrot": "right_heel",
                            "RF": "right_foot"
                            }
smplh2mmm_correspondence = {val: key for key, val in mmm2smplh_correspondence.items()}

smplh2mmm_indexes = [smplh_joints.index(mmm2smplh_correspondence[x]) for x in mmm_joints]

mmm_kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                      [0, 11, 12, 13, 14, 15],
                      [0, 16, 17, 18, 19, 20]]

smplh_to_mmm_scaling_factor = 480 / 0.75
mmm_to_smplh_scaling_factor = 0.75 / 480
mmm_joints = ["root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
              "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"]



smplnh_joints = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee",
                 "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
                 "left_foot", "right_foot", "neck", "left_collar", "right_collar",
                 "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist"]


smplnh2smplh_correspondence = {key: key for key in smplnh_joints}
smplh2smplnh_correspondence = {val: key for key, val in smplnh2smplh_correspondence.items()}

smplh2smplnh_indexes = [smplh_joints.index(smplnh2smplh_correspondence[x]) for x in smplnh_joints]

smplh_to_mmm_scaling_factor = 480 / 0.75
mmm_to_smplh_scaling_factor = 0.75 / 480

mmm_joints_info = {"root": mmm_joints.index("root"),
                   "feet": [mmm_joints.index("LMrot"), mmm_joints.index("RMrot"),
                            mmm_joints.index("LF"), mmm_joints.index("RF")],
                   "shoulders": [mmm_joints.index("LS"), mmm_joints.index("RS")],
                   "hips": [mmm_joints.index("LH"), mmm_joints.index("RH")]}

smplnh_joints_info = {"root": smplnh_joints.index("pelvis"),
                      "feet": [smplnh_joints.index("left_ankle"), smplnh_joints.index("right_ankle"),
                               smplnh_joints.index("left_foot"), smplnh_joints.index("right_foot")],
                      "shoulders": [smplnh_joints.index("left_shoulder"), smplnh_joints.index("right_shoulder")],
                      "hips": [smplnh_joints.index("left_hip"), smplnh_joints.index("right_hip")]}


infos = {"mmm": mmm_joints_info,
         "smplnh": smplnh_joints_info
}

smplh_indexes = {"mmm": smplh2mmm_indexes,
                 "smplnh": smplh2smplnh_indexes}


root_joints = {"mmm": mmm_joints_info["root"],
               "mmmns": mmm_joints_info["root"],
               "smplmmm": mmm_joints_info["root"],
               "smplnh": smplnh_joints_info["root"],
               "smplh": smplh_joints.index("pelvis")
               }

def get_root_idx(joinstype):
    return root_joints[joinstype]


def get_body_model(path, model_type, gender, batch_size, device='cpu', ext='pkl'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    mtype = model_type.upper()
    if gender != 'neutral':
        if not isinstance(gender, str):
            gender = str(gender.astype(str)).upper()
        else:
            gender = gender.upper()
    else:
        gender = gender.upper()
    body_model_path = Path(path) / model_type / f'{mtype}_{gender}.{ext}'

    body_model = smplx.create(body_model_path, model_type=type,
                              gender=gender, ext=ext,
                              use_pca=False,
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=False,
                              create_right_hand_pose=False,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=batch_size)
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c

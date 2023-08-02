from utils.torch import *
from utils.logger import *
from utils.loss import *
from utils.utils import *
import utils.dct

def footskating(prediction):
    # M, B, N, T, V, C = prediction.shape
    z = prediction[:,:,:,1:,[3, 6],1].view(-1).cpu().numpy()
    # M, B, N, T, 2
    idx1 = np.where(z < 0.05)[0]
    deltaxy = torch.norm(prediction[:,:,:,1:,[3, 6],[0,2]] - prediction[:,:,:,:-1,[3, 6],[0,2]], dim=-1).cpu().numpy()
    idx2 = np.where(deltaxy > 0.005)[0]
    idx = np.intersect1d(idx1, idx2)
    return np.shape(idx)[0], np.shape(idx1)[0]


def pred_col(pred):
    """Check Collision between primary prediction and neighbour predictions."""

    def collision(path1, path2, person_radius=0.1, inter_parts=2):
        """Check Collision between path1 and path2.
        path1 = Num_timesteps x 2
        path2 = Num_timesteps x 2
        """
        def getinsidepoints(p1, p2, parts=2):
            """return: equally distanced points between starting and ending "control" points"""

            return np.array((np.linspace(p1[0], p2[0], parts + 1),
                            np.linspace(p1[1], p2[1], parts + 1)))

        for i in range(len(path1) - 1):
            p1, p2 = [path1[i][0], path1[i][1]], [path1[i + 1][0], path1[i + 1][1]]
            p3, p4 = [path2[i][0], path2[i][1]], [path2[i + 1][0], path2[i + 1][1]]
            if np.min(np.linalg.norm(getinsidepoints(p1, p2, inter_parts) - getinsidepoints(p3, p4, inter_parts), axis=0)) \
            <= 2 * person_radius:
                return 1.0
        return 0.0
    
    M, B, N, T, V, C = pred.shape
    col = 0
    pred = pred.cpu().numpy()
    for m in range(M):
        for b in range(B):
            col += collision(pred[m, b, 0, :, 0, [0, 2]], pred[m, b, 1, :, 0, [0, 2]])
            col += collision(pred[m, b, 0, :, 0, [0, 2]], pred[m, b, 2, :, 0, [0, 2]])
            col += collision(pred[m, b, 1, :, 0, [0, 2]], pred[m, b, 2, :, 0, [0, 2]])

    return col / (M * B * 3)

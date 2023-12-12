# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch


class F2(object):
    def __init__(self, lmbda: float):
        super(F2, self).__init__()
        self.lmbda = lmbda

    def penalty(self, x, factors): #TODO: remove x
        norm, raw = 0, 0
        for f in factors:
            raw += torch.sum(f ** 2)
            norm += self.lmbda * torch.sum(f ** 2)
        return norm / factors[0].shape[0], raw / factors[0].shape[0], self.lmbda
    
    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))

class N3(object):
    def __init__(self, lmbda: float):
        super(N3, self).__init__()
        self.lmbda = lmbda

    def penalty(self, x, factors):
        """

        :param factors: tuple, (s, p, o), batch_size * rank
        :return:
        """
        norm, raw = 0, 0
        for f in factors:
            raw += torch.sum(
                torch.abs(f) ** 3
            )
            norm += self.lmbda * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0], raw / factors[0].shape[0], self.lmbda
    
    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))
            
class DURA_UniBi_2(object):
    def __init__(self, lmbda: float):
        super(DURA_UniBi_2, self).__init__()
        self.lmbda = lmbda

    def givens_rotations(self, r, x, transpose=False):
        """Givens rotations.

        Args:
            r: torch.Tensor of shape (N x d), rotation parameters
            x: torch.Tensor of shape (N x d), points to rotate
            transpose: whether to transpose the rotation matrix

        Returns:
            torch.Tensor os shape (N x d) representing rotation of x by r
        """
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], -1, 2))
        if transpose:
            x_rot = givens[:, :, 0:1] * x - givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        else:
            x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_rot.view((r.shape[0], -1))

    def penalty(self, x, factors):
        norm, raw = 0, 0
        h, Rot_u, Rot_v, rel, t = factors
        uh = self.givens_rotations(Rot_u, h)
        suh = rel * uh
        vt = self.givens_rotations(Rot_v, t, transpose=True)
        svt = rel * vt

        norm += torch.sum(suh ** 2 + svt ** 2 + h ** 2 + t ** 2)
        return self.lmbda * norm / h.shape[0], norm / h.shape[0], self.lmbda


    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))


class DURA_UniBi_3(object):
    def __init__(self, lmbda: float):
        super(DURA_UniBi_3, self).__init__()
        self.lmbda = lmbda

    def quaternion_rotation(self, rotation, x, right = False, transpose = False):
        """rotate a batch of quaterions with by orthogonal rotation matrices

        Args:
            rotation (torch.Tensor): parameters that used to rotate other vectors [batch_size, rank]
            x (torch.Tensor): vectors that need to be rotated [batch_size, rank]
            right (bool): whether to rotate left or right
            transpose (bool): whether to transpose the rotation matrix

        Returns:
            rotated_vectors(torch.Tensor): rotated_results [batch_size, rank]
        """
        # ! it seems like calculate in this way will slow down the speed
        # turn to quaterion
        rotation = rotation.view(rotation.shape[0], -1, 4)
        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True).clamp_min(1e-15)  # unify each quaterion of the rotation part
        p, q, r, s = rotation[:, :, 0], rotation[:, :, 1], rotation[:, :, 2], rotation[:, :, 3]
        if transpose:
            q, r, s = -q, -r ,-s # the transpose of this quaterion rotation matrix can be achieved by negating the q, r, s
        s_a, x_a, y_a, z_a = x.chunk(dim=1, chunks=4)
        
        if right:
            # right rotation
            # the original version used in QuatE is actually the right rotation
            rotated_s = s_a * p - x_a * q - y_a * r - z_a * s
            rotated_x = s_a * q + p * x_a + y_a * s - r * z_a
            rotated_y = s_a * r + p * y_a + z_a * q - s * x_a
            rotated_z = s_a * s + p * z_a + x_a * r - q * y_a
        else:
            # left rotation
            rotated_s = s_a * p - x_a * q - y_a * r - z_a * s
            rotated_x = s_a * q + x_a * p - y_a * s + z_a * r
            rotated_y = s_a * r + x_a * s + y_a * p - z_a * q 
            rotated_z = s_a * s - x_a * r + y_a * q + z_a * p 

        rotated_vectors = torch.cat([rotated_s, rotated_x, rotated_y, rotated_z], dim=-1)

        return rotated_vectors

    def penalty(self, x, factors):
        norm, raw = 0, 0
        h, Rot_u, Rot_v, rel, t = factors
        uh = self.quaternion_rotation(Rot_u, h)
        suh = rel * uh
        vt = self.quaternion_rotation(Rot_v, t, transpose=True)
        svt = rel * vt

        norm += torch.sum(suh ** 2 + svt ** 2 + h ** 2 + t ** 2)
        return self.lmbda * norm / h.shape[0], norm / h.shape[0], self.lmbda


    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))
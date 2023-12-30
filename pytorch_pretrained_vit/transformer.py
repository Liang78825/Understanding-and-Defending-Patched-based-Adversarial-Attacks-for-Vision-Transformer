"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch import nn
import torch
from torch import Tensor
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout, layer_id):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.layer_id = layer_id
        self.scores = None  # for visualization

        self.input = None
        self.patch = None
        self.benign_q = None
        self.benign_k = None
        self.feature = None
        self.atten_grad = None
        self.clean = -1
        self.print = -1

    def get_grad(self, X):
        self.atten_grad = X

    def plot_kq(self, k, q, p=None):
        with torch.no_grad():
            plt.figure(dpi=700)
            plt.rcParams["axes.linewidth"] = 2.50
            if False:

                qscore, kscore= (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k])
                # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
                scores = qscore @ kscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                q2 = scores.mean(1)[0,1:,1:].mean(0).max() - scores.mean(1)[0,1:,1:].mean(0) + 0.01
                q2 = q2.view(1,-1)
                k2 = scores.mean(1)[0,1:,1:].mean(0)  - scores.mean(1)[0,1:,1:].mean(0).max()  - 0.1
                k2 = k2.view(1,-1)

                scores_q = qscore @ qscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                scores_k = kscore @ kscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                q1 = scores_q.mean(1)[0,1:,1:].mean(0) + scores_k.mean(1)[0,1:,1:].mean(0)
                q1 = torch.square(q1.max() - q1 + 3) -9
                q1 = q1.view(1, -1)
                k1 = q1


                qscore, kscore = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [self.benign_q, self.benign_k])
                scores = qscore @ kscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                qb2 = scores.mean(1)[0, 1:, 1:].mean(0).max() - scores.mean(1)[0, 1:, 1:].mean(0) + 0.1
                qb2 = qb2.view(1, -1)
                kb2 = scores.mean(1)[0, 1:, 1:].mean(0) - scores.mean(1)[0, 1:, 1:].mean(0).max() - 0.1
                kb2 = kb2.view(1, -1)
                scores_q = qscore @ qscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                scores_k = kscore @ kscore.transpose(-2, -1) / np.sqrt(k.size(-1))
                qb1 = scores_q.mean(1)[0,1:,1:].mean(0) + scores_k.mean(1)[0,1:,1:].mean(0)
                qb1 = torch.square(qb1.max() - qb1+3) - 9
                qb1 = qb1.view(1, -1)
                kb1 = qb1
                filename = 'kq'

            elif True:
                qk = torch.zeros(q.size(1)*2,q.size(2)).cuda()
                qk[0:q.size(1),:] = self.benign_q
                qk[q.size(1):,:] = self.benign_k
                u,a,v = torch.pca_lowrank(qk,q=2)
                q1 = q[:,1:]@v[:,0]
                q2 = q[:,1:] @ v[:, 1]
                k1 = k[:,1:]@v[:,0]
                k2 = k[:,1:]@v[:,1]
                qb1 = self.benign_q[:,1:]@v[:,0]
                qb2 = self.benign_q[:,1:]@v[:,1]
                kb1 = self.benign_k[:,1:]@v[:,0]
                kb2 = self.benign_q[:,1:]@v[:,1]
                filename= 'kq3'

            else:
              #  qv = self.benign_q[:,self.feature].mean(1)
              #  kv = self.benign_k[:,self.feature].mean(1)
                qv = self.benign_q.mean(1)
                kv = self.benign_k.mean(1)
                qv = qv[0] / qv.norm()
                kv = kv[0] / kv.norm()
                kv = kv - qv * torch.dot(kv, qv)
                q1 =  (q[:,1:]@qv).max() - q[:,1:]@(qv)
                q2 = q[:,1:]@(kv)- (q[:,1:]@kv).max()
                k1 = k[:,1:]@(kv)- (k[:,1:]@kv).max()
                k2 =   (k[:,1:]@qv).max()-k[:,1:]@(qv)
                qb1=  (self.benign_q[:,1:]@qv).max() - self.benign_q[:,1:]@(qv)
                qb2 =  self.benign_q[:,1:]@(kv) - (self.benign_q[:,1:]@kv).max()
                kb1 =  self.benign_k[:, 1:] @ (kv) - (self.benign_k[:,1:]@kv).max()
                kb2 =  (self.benign_k[:,1:]@qv).max() -self.benign_k[:, 1:] @ (qv)
                filename = 'kq2'

            qs = torch.cosine_similarity(q[0],self.benign_q[0])
            qs = qs <0.9
            qs[p] = False
            ks = torch.cosine_similarity(k[0],self.benign_k[0])
            ks = ks<0.9
            ks[p] = False

            qs = qs[1:]
            ks = ks[1:]
            p = p-1
            self.feature = self.feature[1:]
            self.benign_q = self.benign_q[1:]
            self.benign_k = self.benign_k[1:]

            # Plot None featured q and k
            plt.scatter(q1[0,torch.logical_and(torch.logical_not(qs), torch.logical_not(self.feature))].cpu().detach().numpy(),
                        q2[0,torch.logical_and(torch.logical_not(qs), torch.logical_not(self.feature))].cpu().detach().numpy(),c='lightsalmon',s=150)
            plt.scatter(k1[0,torch.logical_and(torch.logical_not(ks), torch.logical_not(self.feature))].cpu().detach().numpy(),
                        k2[0,torch.logical_and(torch.logical_not(ks), torch.logical_not(self.feature))].cpu().detach().numpy(),c='lightblue', marker='^',s=150)
            # Plot featured q and k
            plt.scatter(q1[0,torch.logical_and(torch.logical_not(qs), self.feature)].cpu().detach().numpy(),
                        q2[0,torch.logical_and(torch.logical_not(qs), self.feature)].cpu().detach().numpy(),c='firebrick',s=150)
            plt.scatter(k1[0,torch.logical_and(torch.logical_not(ks), self.feature)].cpu().detach().numpy(),
                        k2[0,torch.logical_and(torch.logical_not(ks), self.feature)].cpu().detach().numpy(),c='deepskyblue', marker='^',s=150)

            if qs.sum() != 0:
               # print('query change,', qs.sum())
                qq1 = qb1[0,qs]
                qq2 = qb2[0,qs]
                for i in range(qs.sum()):
                    if self.feature[ks[i]] is True:
                        co = 'firebrick'
                    else:
                        co = 'lightsalmon'

                    plt.arrow(qq1.cpu().detach().numpy()[i], qq2.cpu().detach().numpy()[i],
                              q1[0, qs].cpu().detach().numpy()[i] - qq1.cpu().detach().numpy()[i],
                              q2[0, qs].cpu().detach().numpy()[i] - qq2.cpu().detach().numpy()[i],lw=1.3,ls='--', color= 'firebrick')

                    plt.scatter(q1[0, qs].cpu().detach().numpy()[i], q2[0, qs].cpu().detach().numpy()[i],s=250, c=co,  linewidths=3, edgecolors='greenyellow')

                    plt.scatter(qq1.cpu().detach().numpy()[i], qq2.cpu().detach().numpy()[i], c='none', marker='o', linewidths=2.5,
                                edgecolors='greenyellow',  linestyle='dotted',s=200)

            if ks.sum() != 0:
              #  print('key change,', qs.sum())
                kk1 = kb1[0,ks]
                kk2 = kb2[0,ks]
                for i in range(ks.sum()):
                    if self.feature[ks[i]] is True:
                        co = 'deepskyblue'
                    else:
                        co = 'lightblue'

                    plt.arrow(kk1.cpu().detach().numpy()[i], kk2.cpu().detach().numpy()[i],
                              k1[0, ks].cpu().detach().numpy()[i] - kk1.cpu().detach().numpy()[i],
                              k2[0, ks].cpu().detach().numpy()[i] - kk2.cpu().detach().numpy()[i], lw=1.3, ls='--', color='lightblue')

                    plt.scatter(k1[0, ks].cpu().detach().numpy()[i], k2[0, ks].cpu().detach().numpy()[i] ,marker='^',s=250, c=co,  linewidths=3, edgecolors='greenyellow')

                    plt.scatter(kk1.cpu().detach().numpy()[i], kk2.cpu().detach().numpy()[i], c='none', marker = '^',linewidths=2.5, edgecolors='greenyellow',linestyle='dotted',s=200)


            if p is not None:
                if self.feature[p] is True:
                    co = 'deepskyblue'
                else:
                    co = 'lightblue'

                plt.scatter(k1[0,p].cpu().detach().numpy(), k2[0,p].cpu().detach().numpy(),c=co,marker='^',s=600,linewidths=3, edgecolors='red')
                kk1 = kb1[0,p]
                kk2 = kb2[0,p]

                plt.scatter(kk1.cpu().detach().numpy(), kk2.cpu().detach().numpy(),c='none',marker='^',s=450, linewidths=5, edgecolors='red',linestyle='dotted')
                plt.arrow(kk1.cpu().detach().numpy()[0][0], kk2.cpu().detach().numpy()[0][0],
                          k1[0, p].cpu().detach().numpy()[0][0]- kk1.cpu().detach().numpy()[0][0] ,
                          k2[0, p].cpu().detach().numpy()[0][0]- kk2.cpu().detach().numpy()[0][0], lw=2,ls='--')
            plt.savefig('/home/liang/vit/' +filename + '/'+str(self.print)+'_'+ str(self.layer_id) + '.png')
            plt.close()


    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W

        xx = torch.zeros(1,145,768)
        xx[0,1] = torch.ones(768)
        q, k, v = self.proj_q(xx), self.proj_k(xx), self.proj_v(xx)
        save_image((x - x.min())/ (x.max() - x.min()),'/tmp/pycharm_project_950/x.png')
        """

        self.input = x
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        if self.patch is None:
            self.benign_q = q.clone()
            self.benign_k = k.clone()
        if self.print >=0 and self.layer_id <12:
            #self.print = self.print
            self.plot_kq(k,q,self.patch)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        self.scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            self.scores -= 10000.0 * (1.0 - mask)
        self.scores = self.drop(F.softmax(self.scores, dim=-1))


       # self.scores.register_hook(self.get_grad)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (self.scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout,lay_id):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout,lay_id)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.layer_id = lay_id
        self.x_grad = None
        self.x_grad_1 = None
        self.ratio = None
        self.ratio_avg = None
        self.in_x = None
        self.in_x_1 = None
        self.skip = False
        self.get_grad = -1

    def get_grad_x(self, X):
        self.x_grad = X

    def get_grad_x_1(self, X):
        self.x_grad_1 = X

    def forward(self, x, mask):
        if self.skip:
           # h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
            h = self.drop(self.pwff(self.norm2(x)))
            x = x + h
            return x
        else:
            self.in_x = x.clone()
            self.in_x_1 = self.attn(self.norm1(x), mask)
            h = self.drop(self.proj(self.in_x_1))

              #  self.ratio_avg = aa.mean()

            x1 = x + h
            h = self.drop(self.pwff(self.norm2(x1)))
          #  self.in_x = x.clone()
        #    self.ratio = h.norm/x.norm() #torch.cosine_similarity(x,h).mean()
            x2 = x1 + h
            return x2


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""


    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout, i) for i in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

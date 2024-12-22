import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.optim as optim

class GCN1(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim, dropout, drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.projection = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.dropout = dropout
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(embedding, edge_index)
        return x

    def rep_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout, training=self.training)
        return self.projection(embedding)

    def drop_feature(self, x, drop_prob):
        drop_mask = torch.rand(x.size(1), device=x.device) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def drop_edge(self, edge_index, drop_prob_1, drop_prob_2):
        num_edges = edge_index.size(1)
        mask_1 = torch.rand(num_edges, device=edge_index.device) > drop_prob_1
        mask_2 = torch.rand(num_edges, device=edge_index.device) > drop_prob_2
        edge_index_1 = edge_index[:, mask_1]
        edge_index_2 = edge_index[:, mask_2]
        return edge_index_1, edge_index_2
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def supervised_con_loss(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, t: float):
        f = lambda x: torch.exp(x / t)
        sim = self.sim

        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        neg_mask = ~pos_mask

        pos_sim = between_sim * pos_mask.float()
        neg_sim = between_sim * neg_mask.float()

        pos_sum = pos_sim.sum(1)
        neg_sum = neg_sim.sum(1)

        refl_sum = refl_sim.sum(1) - refl_sim.diag()

        loss = -torch.log(pos_sum / (pos_sum + refl_sum + neg_sum)).mean()

        return loss

    def com_distillation_loss(self, t_logits, s_logits, edge_index, temp):
        s_dist = F.log_softmax(s_logits / temp, dim=-1)
        t_dist = F.softmax(t_logits / temp, dim=-1)
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

        s_dist_neigh = F.log_softmax(s_logits[edge_index[0]] / temp, dim=-1)
        t_dist_neigh = F.softmax(t_logits[edge_index[1]] / temp, dim=-1)

        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

        return kd_loss
    
    def con_loss(self, z1: torch.Tensor, z2: torch.Tensor, t):
        f = lambda x: torch.exp(x / t)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()

    def get_loss(self, pred, true):
        return nn.CrossEntropyLoss()(pred, true)

    def filter_edges_by_mask(self, edge_index, mask):
        mask_indices = mask.nonzero(as_tuple=False).squeeze()
        edge_mask = (edge_index[0].unsqueeze(1) == mask_indices).any(dim=1) & (edge_index[1].unsqueeze(1) == mask_indices).any(dim=1)
        return edge_index[:, edge_mask]

    def forward_two_views(self, x, edge_index):
        x1, x2 = self.drop_feature(x, self.drop_feature_rate_1), self.drop_feature(x, self.drop_feature_rate_2)
        edge_index_1, edge_index_2 = self.drop_edge(edge_index, self.drop_edge_rate_1, self.drop_edge_rate_2)
        x1 = self.conv1(x1, edge_index_1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.projection(x1)
        x2 = self.conv1(x2, edge_index_2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.projection(x2)

        return x1, x2

    def forward_full(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


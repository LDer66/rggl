import argparse
import warnings
import csv
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from util.task_util import accuracy,get_clean_and_noisy_samples
from util.base_util import seed_everything, load_dataset
from model_cr import GCN1
from util.task_util import com_distillation_loss, filter_edges_by_mask,calculate_entropy
from util.plus_noise import add_noise_to_subgraphs
from torch_geometric.utils import to_dense_adj, add_self_loops
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()


# experimental environment setup
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="ogbn-arxiv",choices=["Cora","CiteSeer","PubMed","Computers","Photo","CS","Physics","ogbn-arxiv","Wiki"])
parser.add_argument('--partition', type=str, default="Louvain")
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=30)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_dims', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--mu1', type=float, default=0.5)
parser.add_argument('--mu2', type=float, default=0.0)
parser.add_argument('--num_classes', type=int, default=40)
parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
parser.add_argument('--drop_feature_rate_2', type=float, default=0.6)
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1)
parser.add_argument('--drop_edge_rate_2', type=float, default=0.3)
parser.add_argument('--temperature1', help='temperature of cl', type=float, default=0.5)
parser.add_argument('--w1', type=float, default= 1.0, help='dynatic loss')
parser.add_argument('--w2', type=float, default= 1.0, help='dynatic loss')
parser.add_argument('--Alpha', type=float, default=0.7, help='LP wegiht')
parser.add_argument('--warm-up', type=int, default=5)
parser.add_argument('--lp_prop', help='prop steps of label propagation', type=int, default=15)
parser.add_argument('--threshold2', type=float, default=0.95)
parser.add_argument('--use_prox_loss1', type=bool, default=True, help='whether to use prox loss')
parser.add_argument('--use_prox_loss2', type=bool, default=True, help='whether to use prox loss')
parser.add_argument('--dynamic_loss', type=bool, default=True, help='whether to use Dynamic loss')
parser.add_argument('--dynamic_loss1', type=bool, default=True, help='whether to use Dynamic loss')
parser.add_argument('--lp_pan', type=bool, default=True, help='whether to use LP corr')
parser.add_argument('--entropy_wegiht', type=bool, default=True, help='whether to use entropy_wegiht in server')
parser.add_argument('--only_stage1', type=bool, default=True, help='whether to use eonly_stage1 in noise labels')
parser.add_argument('--entropies', type=bool, default=True, help='whether to use eonly_stage1 in noise labels')
parser.add_argument('--js_weight', type=float, default=0.4)
parser.add_argument('--alpha1', type=float, default=0.0)
parser.add_argument('--alpha2', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=0.6)

##noise
parser.add_argument("--noisy_type", type=str, default="pair",help="uniform, pair")
parser.add_argument("--noisy_rate", type=float, default=0.4)

args = parser.parse_args()


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    js_div = 0.5 * (F.kl_div(F.log_softmax(p, dim=1), m, reduction='batchmean') +
                    F.kl_div(F.log_softmax(q, dim=1), m, reduction='batchmean'))
    return js_div
def entropy_minimization(logits):
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    entropy = torch.sum(p * log_p, dim=1)
    return torch.mean(entropy)

if __name__ == "__main__":
    seed_everything(seed=args.seed)
    dataset = load_dataset(args)
    loss_fn = nn.CrossEntropyLoss()
    args.num_classes = dataset.num_classes
    client_noisy_rates = add_noise_to_subgraphs(dataset.subgraphs, args.noisy_rate, args.noisy_type, args.num_classes)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")

    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2
    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    local_models = [GCN1(feat_dim=subgraphs[client_id].x.shape[1], 
                        hid_dim=args.hid_dim, 
                        out_dim=dataset.num_classes,
                        dropout=args.dropout,
                        drop_feature_rate_1=drop_feature_rate_1,
                        drop_feature_rate_2=drop_feature_rate_2,
                        drop_edge_rate_1=drop_edge_rate_1,
                        drop_edge_rate_2=drop_edge_rate_2).to(device)
                    for client_id in range(args.num_clients)]
    
    local_optimizers = [torch.optim.Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    
    global_model = GCN1(feat_dim=subgraphs[0].x.shape[1], 
                        hid_dim=args.hid_dim, 
                        out_dim=dataset.num_classes,
                        dropout=args.dropout,
                        drop_feature_rate_1=drop_feature_rate_1,
                        drop_feature_rate_2=drop_feature_rate_2,
                        drop_edge_rate_1=drop_edge_rate_1,
                        drop_edge_rate_2=drop_edge_rate_2).to(device)

    best_server_val = 0
    best_server_test = 0
    best_round = 0
    warm_up_done = False
    cached_clean_samples = None
    cached_noisy_samples = None

    for round_id in range(args.num_rounds):

        for client_id in range(args.num_clients):
            local_models[client_id].load_state_dict(global_model.state_dict())

        # global eval
        global_acc_val = 0
        global_acc_test = 0
        global_loss_val = 0
        global_loss_test = 0

        for client_id in range(args.num_clients):
            local_models[client_id].eval()
            
            subgraph = subgraphs[client_id]
            x, edge_index, y = subgraph.x, subgraph.edge_index, subgraph.y
            train_idx, val_idx, test_idx = subgraph.train_idx, subgraph.val_idx, subgraph.test_idx

            with torch.no_grad():
                logits = local_models[client_id].forward_full(x, edge_index)
                loss_val = loss_fn(logits[val_idx], y[val_idx])
                loss_test = loss_fn(logits[test_idx], y[test_idx])
                acc_val = accuracy(logits[val_idx], y[val_idx])
                acc_test = accuracy(logits[test_idx], y[test_idx])

            global_acc_val += subgraph.x.shape[0] / dataset.global_data.x.shape[0] * acc_val
            global_acc_test += subgraph.x.shape[0] / dataset.global_data.x.shape[0] * acc_test
            global_loss_val += subgraph.x.shape[0] / dataset.global_data.x.shape[0] * loss_val.item()
            global_loss_test += subgraph.x.shape[0] / dataset.global_data.x.shape[0] * loss_test.item()

        if global_acc_val > best_server_val:
            best_server_val = global_acc_val
            best_server_test = global_acc_test
            best_round = round_id

        print(f"[server]: current_round: {round_id}\tglobal_val: {global_acc_val:.2f}\tglobal_test: {global_acc_test:.2f}\tloss_val: {global_loss_val:.4f}\tloss_test: {global_loss_test:.4f}")
        print(f"[server]: best_round: {best_round}\tbest_val: {best_server_val:.2f}\tbest_test: {best_server_test:.2f}")
        print("-" * 50)

        if round_id<args.warm_up:
            for client_id in range(args.num_clients):

                for epoch_id in range(args.num_epochs):
                    local_models[client_id].train()
                    global_model.eval()
                    local_optimizers[client_id].zero_grad()
                    subgraph = subgraphs[client_id]
                    x, edge_index, y = subgraph.x, subgraph.edge_index, subgraph.y
                    train_idx = subgraph.train_idx
                    x_o = local_models[client_id].forward(subgraph)
                    supc_loss = loss_fn(x_o[train_idx], y[train_idx])
                    loss = supc_loss
                    if args.use_prox_loss1:
                        w_diff = torch.tensor(0.).to(device)
                        for (local_param, global_param) in zip(local_models[client_id].parameters(), global_model.parameters()):
                            w_diff += torch.pow(torch.norm(global_param - local_param), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += args.mu1  * w_diff
                    loss.backward()
                    local_optimizers[client_id].step()
        else:
            if not warm_up_done or round_id % 1== 0:
                final_clean_samples, noisy_samples = get_clean_and_noisy_samples(subgraphs, local_models, global_model, args, device)
                cached_clean_samples = final_clean_samples
                cached_noisy_samples = noisy_samples
                warm_up_done = True
            else:
                final_clean_samples = cached_clean_samples
                noisy_samples = cached_noisy_samples              
            client_entropies = []
            for client_id in range(args.num_clients):

                clean_sample = torch.tensor(final_clean_samples[client_id], device=device)
                noisy_sample = torch.tensor(noisy_samples[client_id], device=device)

                for epoch_id in range(args.num_epochs):
                    local_models[client_id].train()
                    global_model.eval()
                    local_optimizers[client_id].zero_grad()
                    subgraph = subgraphs[client_id]
                    x, edge_index, y = subgraph.x, subgraph.edge_index, subgraph.y
                    train_idx = subgraph.train_idx
                    x1, x2,x_1,x_2 = local_models[client_id].forward_two_views(x, edge_index)
                    con_loss = 0.5 * (local_models[client_id].con_loss(x1, x2,0.5) + local_models[client_id].con_loss(x2, x1,0.5))

                    x_o = local_models[client_id].forward(subgraph)
                    x_s = global_model.forward(subgraph)
                    clean_idx = torch.where(clean_sample.squeeze())[0]
                    supc_loss_clean = loss_fn(x_o[train_idx][clean_idx], y[train_idx][clean_idx])
                    noisy_idx = torch.where(noisy_sample.squeeze())[0]
                    x_1 = x_1[train_idx][noisy_idx]
                    x_2 = x_2[train_idx][noisy_idx]
                    pred_noisy = (torch.softmax(x_1, dim=1) + torch.softmax(x_2, dim=1)) / 2.0
                    pred_labels = torch.argmax(pred_noisy, dim=1)
                    pred_max = torch.max(pred_noisy, dim=1)[0]    
                    confident_noisy_idx = (pred_max > args.threshold2)
                    assert (confident_noisy_idx >= 0).all() and (confident_noisy_idx < len(pred_labels)).all(), "Confident noisy index out of bounds"
                    #supc_loss_noisy = loss_fn(x_o[train_idx][noisy_idx][confident_noisy_idx], pred_labels[confident_noisy_idx])
                    supc_loss_noisy = (loss_fn(x_1[confident_noisy_idx], pred_labels[confident_noisy_idx])+loss_fn(x_2[confident_noisy_idx], pred_labels[confident_noisy_idx]))/2.0
                    x_o_noisy_confident = x_o[train_idx][noisy_idx][confident_noisy_idx]

                    js_loss = jensen_shannon_divergence(F.softmax(x_1[confident_noisy_idx], dim=1), F.softmax(x_o_noisy_confident, dim=1)) + \
                            jensen_shannon_divergence(F.softmax(x_2[confident_noisy_idx], dim=1), F.softmax(x_o_noisy_confident, dim=1))
                    unconfident_noisy_idx = ~confident_noisy_idx
                    x_o_unconfident = x_o[train_idx][noisy_idx][unconfident_noisy_idx]
                    # entropy_reg_noisy = entropy_minimization(unconfident_samples)
                    
                    supc_loss = supc_loss_clean + supc_loss_noisy * args.beta + js_loss * args.js_weight 
                    loss = con_loss * args.alpha2 + supc_loss
                    if args.use_prox_loss2:
                        w_diff = torch.tensor(0.).to(device)
                        for (local_param, global_param) in zip(local_models[client_id].parameters(), global_model.parameters()):
                            w_diff += torch.pow(torch.norm(global_param - local_param), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += args.mu2  * w_diff
                    loss.backward()
                    local_optimizers[client_id].step()
                with torch.no_grad():
                    local_models[client_id].eval()
                    val_logits = local_models[client_id].forward(subgraphs[client_id])
                    val_entropy = calculate_entropy(val_logits[subgraphs[client_id].val_idx])
                    test_logits = local_models[client_id].forward(subgraphs[client_id])
                    test_entropy = calculate_entropy(test_logits[subgraphs[client_id].test_idx])
                    total_entropy = val_entropy + test_entropy
                    client_entropies.append(total_entropy)
            if args.entropies:
                client_entropies = [1.0 / (entropy + 1e-9) for entropy in client_entropies]

        # global aggregation
        with torch.no_grad():
            if round_id <args.warm_up:
                for client_id in range(args.num_clients):
                    weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] 
                    for (local_state, global_state) in zip(local_models[client_id].parameters(), global_model.parameters()):
                        if client_id == 0:
                            global_state.data = weight * local_state
                        else:
                            global_state.data += weight * local_state
            else:
                if args.entropy_wegiht:
                    client_weight = np.ones(args.num_clients)
                    client_weight = client_weight * client_entropies
                    client_weight = client_weight / client_weight.sum()
                    for client_id in range(args.num_clients):
                        weight = client_weight[client_id]
                        for (local_state, global_state) in zip(local_models[client_id].parameters(), global_model.parameters()):
                            if client_id == 0:
                                global_state.data = weight * local_state
                            else:
                                global_state.data += weight* local_state
                else:
                    for client_id in range(args.num_clients):
                        weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] 
                        for (local_state, global_state) in zip(local_models[client_id].parameters(), global_model.parameters()):
                            if client_id == 0:
                                global_state.data = weight * local_state
                            else:
                                global_state.data += weight * local_state
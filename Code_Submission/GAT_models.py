from torch_geometric.nn import GATConv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.lin1 = torch.nn.Linear(args.input_size, args.hidden_size)
        self.convs = torch.nn.ModuleList()
        for i in range(args.stack_layer_num):
            self.convs.append(
                GATConv(args.hidden_size, args.hidden_size // args.heads, heads=args.heads, dropout=args.att_dropout))

        self.lin3 = torch.nn.Linear(args.hidden_size, 1)
        glorot(self.lin3.weight)
        glorot(self.lin1.weight)

    def forward(self, args, word_embed_matrix, target_mask_list, graph_edge_list, label_id_list=None):
        if args.embed_dropout > 0:
            word_embed_matrix = F.dropout(word_embed_matrix, p=args.embed_dropout, training=self.training)
        word_embed_matrix = self.lin1(word_embed_matrix)

        for i in range(args.stack_layer_num):
            word_embed_matrix = F.elu(self.convs[i](word_embed_matrix, graph_edge_list))
        #endfor

        batch_size = len(target_mask_list)

        target_embed_list = []
        for i in range(batch_size):
            t_mask_i = target_mask_list[i]
            target_embed = word_embed_matrix[t_mask_i]
            target_embed_list.append(target_embed)
        # endfor

        target_embed_list = torch.cat(target_embed_list, dim=0)

        logits = self.lin3(target_embed_list)
        total_score = torch.squeeze(logits, dim=1)

        loss = None
        if label_id_list is not None:
            # positive
            pos_mask = label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        #endif

        return total_score, loss


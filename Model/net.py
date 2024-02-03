# Neural network part, add attention mechanism
from torch import nn
from torch.nn import MultiheadAttention
from torchtuples import tuplefy
from torchtuples.practical import DenseVanillaBlock
import torch.nn.functional as F
class ATT_M(nn.Module):

    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 num_heads=1, attention_dropout=0.0,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()

        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]

        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        self.self_attention = MultiheadAttention(embed_dim=1, num_heads=num_heads, dropout=attention_dropout)
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        x = self.net(input)
        attention_output, _ = self.self_attention(x, x, x)
        x = F.relu(x + attention_output)
        return x
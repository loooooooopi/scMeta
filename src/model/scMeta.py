import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import TransformerConv, GATConv, SAGEConv


class scMeta(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, conv_type='TransformerConv', heads=4, dropout=0.3):
        super(scMeta, self).__init__()
        
        self.conv_type = conv_type
        
        if conv_type == 'TransformerConv':
            self.conv1 = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = TransformerConv(in_channels=hidden_dim * heads, out_channels=hidden_dim, heads=1, dropout=dropout)
            
        elif conv_type == 'GATConv':
            self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = GATConv(in_channels=hidden_dim * heads, out_channels=hidden_dim, heads=1, dropout=dropout)
            
        elif conv_type == 'SAGEConv':
            self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim * heads)
            self.conv2 = SAGEConv(in_channels=hidden_dim * heads, out_channels=hidden_dim)
            self.dropout = Dropout(dropout)
            
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        self.classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, return_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        if self.conv_type == 'SAGEConv':
            x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        logits = self.classifier(x)
        
        if return_embedding:
            return logits, x
        else:
            return logits

class scMetaMLP(torch.nn.Module):
    """Plain per-cell MLP baseline with no message passing at all (edge_index
    is ignored). Matches scMeta's layer widths/heads so capacity is comparable,
    used to test whether the graph structure contributes anything beyond what
    the same features give a cell-independent classifier (Reviewer 2, point 2).
    """
    def __init__(self, input_dim, hidden_dim, num_classes, heads=4, dropout=0.3):
        super(scMetaMLP, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim * heads)
        self.fc2 = Linear(hidden_dim * heads, hidden_dim)
        self.dropout = Dropout(dropout)
        self.classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index=None, return_embedding=False):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)

        if return_embedding:
            return logits, x
        else:
            return logits


def NT_Xent(embeddings, tau=0.5):
    device = embeddings.device
    z_i = F.normalize(embeddings, dim=1)
    z_j = F.normalize(embeddings[torch.randperm(z_i.size(0))], dim=1)

    logits = torch.mm(z_i, z_j.t()) / tau
    labels = torch.arange(z_i.size(0), device=device)
    loss = F.cross_entropy(logits, labels)
    return loss


def graph_nt_xent(embeddings, edge_index, tau=0.5, max_pairs=200_000):
    """Edge-aware contrastive loss: for each real graph edge (src, dst), dst
    should be the most similar node to src among all other nodes in the batch
    (in-batch softmax, i.e. all non-neighbors act as negatives). Unlike
    NT_Xent above -- which pairs each embedding with a *randomly permuted*
    embedding and is therefore blind to edge_index entirely -- this makes the
    contrastive objective actually depend on graph topology, so training can
    reward the model for exploiting real neighbourhoods rather than adding a
    topology-agnostic regularizer that happens to run inside a GNN.
    """
    device = embeddings.device
    if edge_index.size(1) == 0:
        return embeddings.new_zeros(())

    src, dst = edge_index[0], edge_index[1]
    if src.size(0) > max_pairs:
        perm = torch.randperm(src.size(0), device=device)[:max_pairs]
        src, dst = src[perm], dst[perm]

    z = F.normalize(embeddings, dim=1)
    logits = torch.mm(z[src], z.t()) / tau  # [n_pairs, N] -- similarity of each src to every node in batch
    loss = F.cross_entropy(logits, dst)
    return loss



from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.scaling_factor = float(key_dim) ** -0.5
        self.dot = nn.Linear(query_dim, key_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout()
        self.weight = nn.Linear(key_dim, key_dim)
 
    def forward(self, query, key, value, mask_bool=False, dropout_p=0):
        mask = None
        if mask_bool:
            lengths = mask.sum(dim=1)
            max_len = lengths.max().long()
            indices = mask.nonzero()
            indices = indices[:, :, 0] * max_len + indices[:, :, 1]
            indices = indices.unsqueeze(2)
            indices = indices.expand(-1, -1, query.shape[-1])
            mask, _ = indices.sort(dim=1)
            mask, _ = mask.reshape(-1)
            mask = mask[:max_len**2].clone()
            mask = mask.unsqueeze(0)
            mask = mask.expand(query.shape[0], -1)
        score = self.softmax(self.scaling_factor * (query + self.dot(key)).sum(-1))
        if dropout_p > 0:
            score = self.dropout(score)
        score = score.unsqueeze(1).expand(-1, key.shape[1], -1)
        out = (score * value).sum(dim=-1)
        if mask is not None:
            result = out.clone()
            result[mask] = 0
            out = result
        return out

# Initializing the model
query_dim = 10
key_dim = 5
attn = ScaledDotProductAttention(query_dim, key_dim)

# Inputs to the model
query = torch.randn(1, 3, query_dim)
key = torch.randn(1, 5, key_dim)
value = torch.randn(1, 5, key_dim)
mask = torch.ByteTensor([[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]).view(1, 3, 5)
dropout_p = 0.2


import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(64, 64)
        self.k = torch.nn.Linear(64, 64)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk.div(64.**0.5)
        attn = F.softmax(qk, dim=-1)
        if self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        return attn.matmul(value)
 
    def reset_parameters():
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.xavier_uniform_(self.k.weight)

# Initializing the model
m = Model()
m.dropout_p = 0.0

# Inputs to the model
query = torch.randn(1, 128, 64)
key = torch.randn(1, 25, 64)
value = torch.randn(1, 25, 64)

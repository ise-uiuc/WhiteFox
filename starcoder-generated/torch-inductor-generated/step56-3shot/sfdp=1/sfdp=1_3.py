
from torch import nn

class Model(nn.Module):
    def __init__(self, d_q, d_v, n_heads, dropout_p):
        super().__init__()
        self.key = nn.Linear(d_v, d_k)
        self.dropout = nn.Dropout(dropout_p)
 
    def forward(self, x1, x2, x3):
        k = self.key(x2)
        dot = torch.matmul(x1, k.transpose(-2, -1))
        inv_scale = 1.0 / math.sqrt(math.sqrt(d_k))
        scaled = dot * inv_scale
        softmax = torch.softmax(scaled, -1)
        dropout = self.dropout(softmax)
        output = torch.matmul(dropout, x3)
        return output

# Initializing the model
d_model = 1024
n_heads = 16
dropout_p = 0.1
d_k = math.sqrt(d_model)
d_v = d_k
m = Model(d_k, d_v, n_heads, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 4, 1024)
x2 = torch.randn(1, 4, 1024)
x3 = torch.randn(1, 4, 1024)

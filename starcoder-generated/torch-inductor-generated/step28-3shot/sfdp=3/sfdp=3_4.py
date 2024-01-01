
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
    
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
dropout_p = 0.5
b = 1
c = 4
m = Model(dropout_p)

# Inputs to the model
query = torch.randn(b, 1, 28, 28)
key = torch.randn(b, 4, 14, 14)
value = torch.randn(b, 4, 14, 14)
scale_factor = query.size(-1)**0.5


from torch import nn

class Model(nn.Module):
    def __init__(self, dropout_p):    
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v, scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = scale_factor.pow(-1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.3
m = Model(dropout_p)

# Inputs to the model
q = torch.randn(32, 10, 64)
k = torch.randn(32, 16, 64)
v = torch.randn(32, 16, 64)
scale_factor = torch.randint(low=1, high=256, size=[32, 1])

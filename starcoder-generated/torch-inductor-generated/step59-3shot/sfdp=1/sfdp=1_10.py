
import torch.nn.functional as F
 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 4, 64)
key = torch.randn(2, 6, 64)
value = torch.randn(2, 6, 64)
inv_scale_factor = torch.reciprocal(torch.arange(value.numel(), out=torch.FloatTensor()).fill_(4))
dropout_p = 0.1

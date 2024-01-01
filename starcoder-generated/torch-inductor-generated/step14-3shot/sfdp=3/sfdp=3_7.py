
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, n_in1, n_in2, n_out, dropout_p):
        super().__init__()
        self.linear = nn.Linear(n_in1, n_out)
        self.dropout = nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = (n_in2**-.5)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = self.dropout(dropout_qk).matmul(value)
        return output

# Initializing the model
n_in1, n_in2, n_out = 3, 6, 1
dropout_p = 0.2
m = Attention(n_in1, n_in2, n_out, dropout_p)

# Inputs to the model
query = torch.normal(n_in1, n_in2)
key = torch.normal(n_in1, n_in2)
value = torch.normal(n_in2, n_in2)

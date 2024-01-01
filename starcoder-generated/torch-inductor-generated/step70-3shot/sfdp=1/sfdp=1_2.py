
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_heads = 1
        d_model = 32
        dropout_rate = 0
 
    def forward(self, q, k, v, mask, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
N = 3
L = 5
C = 32
m = Model()

# Inputs to the model
q = torch.randn(L, C)
k = torch.randn(N, L, C)
v = torch.randn(N, L, C)
mask = torch.randn_like(q)
inv_scale_factor = 1.0
dropout_pt = 0.0

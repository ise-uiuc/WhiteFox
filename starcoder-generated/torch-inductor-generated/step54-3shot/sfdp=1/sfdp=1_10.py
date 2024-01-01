
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, dropout_p, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        o = dropout_qk.matmul(v)
        return o

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 32, 64)
k = torch.randn(1, 8, 64, 128)
v = torch.randn(1, 8, 64, 128)
dropout_p = 0.
inv_scale_factor = 1.

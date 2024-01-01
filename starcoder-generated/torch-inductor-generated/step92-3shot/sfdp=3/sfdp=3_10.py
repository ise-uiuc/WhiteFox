
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1 / np.power(float(k.size(-1)), 0.5)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(v)
        return output, dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2, 10, 64)
k = torch.randn(1, 2, 20, 64)
v = torch.randn(1, 2, 20, 64)
_, __output1__ = m(q, k, v)

# Inputs to the model
q = torch.randn(1, 2, 10, 64)
k = torch.randn(1, 2, 20, 64)
v = torch.randn(1, 2, 20, 64)
__output2__, _ = m(q, k, v)


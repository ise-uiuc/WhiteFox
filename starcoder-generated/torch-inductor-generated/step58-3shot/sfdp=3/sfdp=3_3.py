
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        sf = qk.mul(scale_factor)
        softmax_qk = sf.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 1, 2, 3)
k = torch.randn(2, 1, 3, 4)
v = torch.randn(2, 1, 3, 4)
output = m(q, k, v)



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(k.size(-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 4, 150)
k = torch.randn(3, 5, 150)
v = torch.randn(3, 5, 200)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1)) / math.sqrt(x1.size(-1))
        qk = qk + x3
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(x3, x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4, 5)
x2 = torch.randn(3, 5, 6)
x3 = torch.randn(3, 4, 6)

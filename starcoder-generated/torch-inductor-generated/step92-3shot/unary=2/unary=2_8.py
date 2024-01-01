
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.transpose(x1, 1, 0)
        v2 = v1.transpose(1, 0)
        v3 = torch.transpose(v2, 1, 0)
        v4 = v3.transpose(1, 0)
        v5 = torch.transpose(v4, 0, 2)
        v6 = v5.transpose(0, 2)
        v7 = v6.transpose(0, 2)
        v8 = torch.transpose(v7, 1, 0)
        v9 = v8.transpose(1, 0)
        return v9
# Inputs to the model
x1 = torch.randn(4, 31, 28)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = torch.transpose(v1, 1, 2)
        v3 = torch.transpose(x1, -2, -1)
        return v1, v2, v3
# Inputs to the model
x1 = torch.randn(2, 3, 5, 7)

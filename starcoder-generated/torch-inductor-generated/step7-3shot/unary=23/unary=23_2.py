
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.transpose(x1, 0, 1)
        v2 = torch.transpose(x1, 1, 2)
        v3 = torch.transpose(x1, 2, 3)
        return v3
# Inputs to the model
x1 = torch.randn(3, 4, 5, 5)

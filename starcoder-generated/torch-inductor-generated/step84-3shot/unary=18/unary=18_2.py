
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.add(x1, x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
x2 = torch.randn(1, 2, 16, 16)

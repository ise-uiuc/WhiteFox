
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1):
        v1 = torch.max(x1)
        return v1
# Inputs to the model
x1 = torch.randn(84, 47, 74)

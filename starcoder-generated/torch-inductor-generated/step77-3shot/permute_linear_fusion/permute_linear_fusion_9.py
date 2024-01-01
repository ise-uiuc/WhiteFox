
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.max(x1)
# Inputs to the model
x1 = torch.randn(1, 10, 4)

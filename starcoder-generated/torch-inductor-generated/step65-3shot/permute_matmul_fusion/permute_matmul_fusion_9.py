
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1, x2):
        return x0.permute(0, 2, 1)
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

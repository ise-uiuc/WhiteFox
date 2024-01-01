
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        return x2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(2, 3, 4)
x2 = torch.randn(1, 4, 1)

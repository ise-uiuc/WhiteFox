
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = 10 * x
        x = torch.split(x, [1], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 9)

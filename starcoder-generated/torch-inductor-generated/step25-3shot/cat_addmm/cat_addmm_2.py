
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.stack((x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(3, 2)

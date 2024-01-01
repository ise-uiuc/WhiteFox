
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.stack((torch.cat([x, x], dim=1), torch.cat([x, x], dim=2)), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)

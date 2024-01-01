
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)

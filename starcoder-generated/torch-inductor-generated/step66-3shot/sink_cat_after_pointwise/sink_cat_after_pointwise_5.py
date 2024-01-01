
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.cat([x, x], dim=2)
        return x1.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x], dim=2)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)

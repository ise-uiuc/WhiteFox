
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        y = (torch.cat([y, y], dim=1)).tanh()
        x = y
        return x
# Inputs to the model
x = torch.randn(1, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return x.tanh()
# Inputs to the model
x = torch.randn(2, 3)
y = torch.randn(2, 3)

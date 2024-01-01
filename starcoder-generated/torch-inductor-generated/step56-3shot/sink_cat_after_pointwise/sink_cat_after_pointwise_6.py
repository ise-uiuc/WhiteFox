
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1) if x.shape == (4, 12) else x.relu()
        x = torch.cat([y, y], dim=2)
        return x.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)

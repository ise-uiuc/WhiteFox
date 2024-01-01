
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x * 2
        z = z * 3
        x = torch.cat([x, z], dim=1)
        return x.view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(5, 3, 4)

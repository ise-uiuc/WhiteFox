
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = torch.rand(x.shape[0], 2)
        return torch.cat([v, v, v, v, v, v, v, v, v, v, v, v], 1)
# Inputs to the model
x = torch.randn(32, 224)

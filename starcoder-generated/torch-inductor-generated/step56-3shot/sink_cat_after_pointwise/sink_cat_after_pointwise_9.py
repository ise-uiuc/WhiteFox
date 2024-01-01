
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([x.view(x.shape[0], -1), x.view(x.shape[0], -1)], dim=-1)
# Inputs to the model
x = torch.randn(2, 3, 4)

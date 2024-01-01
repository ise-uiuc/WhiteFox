
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([torch.squeeze(x, dim=1), x], dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 1, 4)

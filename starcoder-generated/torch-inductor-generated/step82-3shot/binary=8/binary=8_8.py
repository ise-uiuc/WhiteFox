
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat([x[..., :3] - x[..., 3:], x[..., :3] + x[..., 3:]], dim=1)
        return v1
# Inputs to the model
x = torch.randn(1, 6, 64, 64)

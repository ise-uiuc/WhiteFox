
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=1.7, recompute_scale_factor=None)
        v1 = v1.sum(dim=1).unsqueeze(dim=1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 5, 6, 4)

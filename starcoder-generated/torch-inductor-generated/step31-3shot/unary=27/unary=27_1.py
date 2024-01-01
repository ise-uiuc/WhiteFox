
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.sum(dim=1, keepdim=True)
        v2 = torch.clamp_min(v1, 0.1)
        v3 = torch.clamp_max(v2, 0.2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 17, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 3
        v2 = v1.clamp_min(0)
        v3 = v2.clamp_max(6)
        v4 = v3.div(6)
        v5 = v4 * 4
        v6 = v5 + 3
        v7 = v6.clamp_min(0)
        v8 = v7.clamp_max(6)
        v9 = v8.div(6)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

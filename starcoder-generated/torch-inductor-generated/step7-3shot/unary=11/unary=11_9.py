
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.add(8)
        v2 = v1.transpose(0, 1)
        v3 = v2.clamp_min(4)
        v4 = v3.clamp_max(v2)
        v5 = v4.div(4)
        return v5
# Inputs to the model
x1 = torch.randn(3, 4, 5, 5)

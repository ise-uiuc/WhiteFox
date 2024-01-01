
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = (19 + 19) + (39 + 39)
        v2 = (19 + 19) * (39 + 39)
        v3 = (19 * 19) + (39 * 39)
        return v1, v2, v3
# Inputs to the model
x1 = torch.randn(2, 1, 14, 16)

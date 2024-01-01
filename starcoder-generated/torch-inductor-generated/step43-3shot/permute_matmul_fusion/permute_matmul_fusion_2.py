
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=[2, 1])
        return v1
# Inputs to the model
x1 = torch.randn(10, 7, 5)

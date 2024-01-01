
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=5.1711, mode='nearest')
        v2 = torch.nn.functional.interpolate(v1, scale_factor=21.1236, mode='bilinear')
        v3 = torch.nn.functional.interpolate(v2, scale_factor=0.1907)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 13, 13)

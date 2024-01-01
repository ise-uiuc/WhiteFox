
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=1.0, size=None, mode='bicubic', align_corners=None)
        v2 = torch.nn.functional.interpolate(x2, scale_factor=2.0, size=None, mode='bicubic', align_corners=None)
        v3 = torch.nn.functional.interpolate(x3, scale_factor=3.0, size=None, mode='bicubic', align_corners=None)
        v4 = torch.nn.functional.interpolate(x4, scale_factor=1.0, size=None, mode='bicubic', align_corners=None)
        v5 = torch.nn.functional.interpolate(x5, scale_factor=1.0, size=None, mode='bicubic', align_corners=None)
        v11 = v1 + v2 + v3
        v12 = v11 + v4 + v5
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32, requires_grad = True)
x2 = torch.randn(1, 16, 32, 32, requires_grad = True)
x3 = torch.randn(1, 16, 32, 32, requires_grad = True)
x4 = torch.randn(1, 16, 32, 32, requires_grad = True)
x5 = torch.randn(1, 16, 32, 32, requires_grad = True)
# model ends

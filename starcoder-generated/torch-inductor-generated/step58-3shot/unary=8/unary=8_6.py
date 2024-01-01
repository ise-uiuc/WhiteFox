
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1):
        v1 = torch.ops.aten.pixel_shuffle2(x1, upscale_factor=2)
        v2 = torch.ops.aten.add(v1, 3, alpha=1)
        v3 = torch.ops.aten.clamp(v2, min=0)
        v4 = torch.ops.aten.clamp(v3, max=6)
        v5 = torch.ops.aten.mul(v1, v4)
        v6 = torch.ops.aten.div(v5, 6, rounding_mode='trunc')
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 64, 32)

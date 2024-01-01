
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.conv2d(x1, torch.ones(8, 3, 1, 1), padding=1, groups=8, bias=None)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = torch.div(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

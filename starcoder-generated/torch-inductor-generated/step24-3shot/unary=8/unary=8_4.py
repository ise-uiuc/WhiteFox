
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.interpolate(x1, mode='bilinear', align_corners=False)
        x2 = torch.relu(t1)
        t3 = x2.clone()
        v4 = torch.nn.functional.interpolate(t3, size=[64], mode='bilinear', align_corners=False)
        v5 = x2 / v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

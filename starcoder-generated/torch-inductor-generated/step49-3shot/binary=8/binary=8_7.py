
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        a = x2[..., :1,...].clone()
        b = x1[..., :1,...].clone()
        c = x1[..., 1:,...].clone()
        d = x2[..., 1:,...].clone()
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(torch.cat([a, v2, c, d], dim=1))
        v4 = v1 + v3
        if torch.onnx.is_in_onnx_export():
            v4 = torch.tanh(x1)
        v5 = v4 - x1
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 16, 16)

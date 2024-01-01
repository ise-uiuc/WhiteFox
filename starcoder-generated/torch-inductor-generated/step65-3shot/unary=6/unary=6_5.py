
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(2, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = (v1 / torch.sqrt(torch.abs(torch.mean(v1, dim=1, keepdim=True)*torch.mean(v1, dim=2, keepdim=True)*torch.mean(v1, dim=3, keepdim=True))))
        v3 = 3.0 + v2
        v4 = v2
        v5 = 0.0
        v6 = 6.0
        v7 = v6 < v5
        v8 = v7.type(dtype=torch.float32)
        v9 = v3 + v8
        v10 = v3 * v8
        return v7
# Inputs to the model
x = torch.randn(1, 2, 4, 8, requires_grad=False)

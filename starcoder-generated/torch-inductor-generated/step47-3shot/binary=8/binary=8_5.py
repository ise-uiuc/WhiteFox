
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v1)
        v5 = self.bn2(v2)
        v6 = torch.sin(v3 + v4 + v5)
        v7 = torch.cos(v3 + v4 + v5)
        v8 = torch.relu(v3 + v4 + v5)
        v9 = v6.mul(v7).div(v8).pow(v3 + v4 - v5)
        v10 = torch.abs(v3 + v4 + v5)
        v11 = v9.div(v10).mul(v4 + v5)
        v12 = torch.clamp(v4 + v5, max=3, min=1).mul(v11)
        v13 = torch.ceil(v12 + 0.5 + v5).sub(0.4 + v5).clamp(-1.0 + v2, 2)
        v14 = v13.reciprocal().clamp(min=1, max=2)
        v15 = v13.neg().addcmul(0.5, v14, value=0.22 + v5).sign()
        v16 = v15.sub(v13.mul(v12)).div(v15.add(v12)).tanh().add(1 + v2)
        v17 = v11.sub(0.4 + v5).div(v6).mul(v3.pow(0.5 + v16) - v7).floor()
        return v17.neg().sub(1 + v18)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.bn5 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x2
        v4 = self.bn2(v3)
        v5 = self.conv3(x1)
        v6 = self.conv4(x2)
        v7 = v5 + v6
        v8 = self.bn4(v7)
        v9 = torch.tanh(self.bn5(v4 + v8))
        v10 = v9.squeeze(0).narrow(0, 0, v9.size(0)).squeeze(0)
        v11 = v10.permute(2, 0, 1)
        s1 = torch.softmax(v11.float().div(1. / 255), dim=2).mul(255).type_as(x1).unsqueeze(-1)
        x2 = x1.permute(2, 0, 1).mul(255).type_as(x2)
        s2 = torch.softmax(x2.float().div(1. / 255), dim=2).mul(255).type_as(x2)
        v12 = torch.tanh(s1.sum(0).mul(s2.sum(0)))
        return v9, v12
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)

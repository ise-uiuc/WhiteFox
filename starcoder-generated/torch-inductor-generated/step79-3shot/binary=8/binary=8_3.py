
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv3(x1)
        v3 = v2 + v1
        v4 = self.relu1(v3)
        v5 = self.conv2(v4)
        v6 = self.conv4(v4)
        v7 = v6 + v5
        v8 = self.conv5(x1)
        v9 = self.conv7(x1)
        v10 = v9 + v8
        v11 = self.conv6(v10)
        v12 = self.conv8(v10)
        s1 = v3.unsqueeze(0) * v11.unsqueeze(0).transpose(0, 2)
        (n, k) = s1.size()[-2:]
        s2 = s1.reshape(n, k, -1).sum(-1).div(k)
        s3 = v10.unsqueeze(0) * v12.unsqueeze(0).transpose(0, 2)
        (n, k) = s3.size()[-2:]
        s4 = s3.reshape(n, k, -1).sum(-1).div(k)
        return (s2, s4)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)

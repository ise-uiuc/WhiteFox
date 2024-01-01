
class Model(torch.nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 5, 15, stride=2, padding=0)
            self.conv2 = torch.nn.Conv2d(5, 10, 11, stride=1, padding=5)
            self.conv3 = torch.nn.Conv2d(5, 10, 11, stride=1, padding=5)
            self.conv4 = torch.nn.Conv2d(10, 20, 7, stride=1, padding=0)
            self.conv5 = torch.nn.Conv2d(20, 50, 7, stride=1, padding=0)
            self.conv6 = torch.nn.Conv2d(50, 100, 7, stride=1, padding=0)
    def forward(self, x1):
            v1 = self.conv1(x1)
            v1 = torch.relu(v1)
            v1 = v1.type(torch.float16)
            v2 = self.conv2(v1)
            v2 = torch.relu(v2)
            v2 = v2.type(torch.float16)
            v3 = self.conv3(v2)
            v3 = torch.relu(v3)
            v3 = v3.type(torch.float16)
            v4 = self.conv4(v3)
            v4 = torch.relu(v4)
            v4 = v4.type(torch.float16)
            v5 = self.conv5(v4)
            v5 = torch.relu(v5)
            v5 = v5.type(torch.float16)
            v6 = self.conv6(v5)
            v6 = torch.relu(v6)
            v6 = v6.type(torch.float16)
            v6 = v6 * 12
            return v6 + 50
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

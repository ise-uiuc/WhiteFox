
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3.unsqueeze(1)
        v5 = self.conv4(v4)
        v6 = v5.squeeze(1)
        v7 = v1.transpose(1, 2)
        v8 = self.conv1(v7)
        v9 = v8.transpose(1, 2)
        v10 = v6 + v9 
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)

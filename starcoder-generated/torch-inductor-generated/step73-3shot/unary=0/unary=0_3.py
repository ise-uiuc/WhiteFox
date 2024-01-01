
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 7, 3, stride=3, padding=3)
        self.conv2 = torch.nn.Conv2d(7, 5, 3, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 4, 3, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 5, 3, stride=3, padding=1)
        self.conv5 = torch.nn.Conv2d(5, 3, 3, stride=3, padding=1)
    def forward(self, x0):
        v1 = self.conv1(x0)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = v5 * 0.5
        v7 = v5 * v5
        v8 = v7 * v5
        v9 = v8 * 0.044715
        v10 = v5 + v9
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v6 * v13
        return v14
# Inputs to the model
x0 = torch.randn(1, 12, 256, 256)

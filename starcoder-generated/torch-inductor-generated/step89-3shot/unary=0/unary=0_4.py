
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 44, 1, stride=14)
        self.conv2 = torch.nn.Conv2d(44, 49, 1, stride=25, padding=19, groups=3)
        self.conv3 = torch.nn.Conv2d(49, 2, 1, stride=27)
        self.conv4 = torch.nn.Conv2d(2, 15, 1, stride=9, padding=6)
    def forward(self, x30):
        v1 = self.conv1(x30)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return self.conv4(self.conv3(self.conv2(v10)))
# Inputs to the model
x30 = torch.randn(1, 4, 211, 276)

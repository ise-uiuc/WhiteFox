
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 3, 7, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 2, 3, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(6, 1, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x3):
        v1 = self.conv7(x3)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x3 = torch.randn(9, 14, 4, 4)

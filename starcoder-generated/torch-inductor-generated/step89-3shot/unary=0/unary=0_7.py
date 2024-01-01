
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 49, 5, stride=3, padding=27)
        self.conv2 = torch.nn.Conv2d(49, 43, 1, stride=7, padding=16)
    def forward(self, x32):
        v1 = self.conv1(x32)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return self.conv2(v10)
# Inputs to the model
x32 = torch.randn(1, 11, 24, 30)

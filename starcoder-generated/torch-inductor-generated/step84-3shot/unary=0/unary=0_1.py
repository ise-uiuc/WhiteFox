
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 2, stride=5, padding=5)
        self.conv2 = torch.nn.Conv2d(3, 1, 2, stride=4, padding=4)
    def forward(self, x546):
        v1 = self.conv1(x546)
        v3 = self.conv2(v1)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x546 = torch.randn(1, 1, 13, 16)

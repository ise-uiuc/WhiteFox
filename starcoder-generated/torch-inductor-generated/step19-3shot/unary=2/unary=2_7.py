
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad14 = torch.nn.ReflectionPad2d((7, 7, 2, 2))
        self.conv14 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.pad14(x1)
        v2 = self.conv14(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        return v10
# Inputs to the model
x1 = torch.randn(2, 1, 32, 16)

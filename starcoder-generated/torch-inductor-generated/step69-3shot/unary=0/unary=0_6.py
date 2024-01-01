
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_layer = torch.nn.ConstantPad2d((0, 0, 0, 0), 0.5231139042069507)
        self.conv = torch.nn.Conv2d(64, 31, 2, stride=1, padding=0)
    def forward(self, x78):
        v1 = self.pad_layer(x78)
        v2 = self.conv(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
x78 = torch.randn(1, 64, 2, 1)

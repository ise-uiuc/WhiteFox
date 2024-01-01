
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 3, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(12, 3, 1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = v1 * 0.5
        v4 = v1 * v1 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        v11 = torch.cat((v2, v10), 1)
        v12 = self.conv(v11)
        return v12
# Inputs to the model
x1 = torch.randn(2, 3, 5, 5)

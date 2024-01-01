
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.padding = 3
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 20, 4, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = F.pad(v1, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        v3 = self.conv_transpose(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3 * v3
        v6 = v5 * 0.044715
        v7 = v3 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v4 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 6, 32, 16)

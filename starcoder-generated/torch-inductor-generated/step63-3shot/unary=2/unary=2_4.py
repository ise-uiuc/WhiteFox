
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv_transpose(v1)
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
x1 = torch.randn(3, 1, 1, 1)

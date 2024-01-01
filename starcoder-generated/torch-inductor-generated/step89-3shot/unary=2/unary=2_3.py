
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 1, padding=1)
        torch.nn.init.zeros_(self.conv_transpose.weight)
    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        v1 = t1 * 0.5
        v2 = t1 * t1 * t1
        v3 = v2 * 0.044715
        v4 = t1 + v3
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = t1 * v7
        return v8
# Inputs to the model
x1 = torch.zeros(3, 4, 2, 5)

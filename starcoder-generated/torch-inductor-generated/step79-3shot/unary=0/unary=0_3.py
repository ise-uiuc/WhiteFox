
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 25, 25, stride=25, padding=0)
    def forward(self, x333):
        v13 = self.conv(x333)
        v14 = v13 * 0.5
        v15 = v13 * v13
        v16 = v15 * v13
        v17 = v16 * 0.044715
        v18 = v13 + v17
        v19 = v18 * 0.7978845608028654
        v20 = torch.tanh(v19)
        v21 = v20 + 1
        v22 = v14 * v21
        return v22
# Inputs to the model
x333 = torch.randn(1, 1, 25, 25)

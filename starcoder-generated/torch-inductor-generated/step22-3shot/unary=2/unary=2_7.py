
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(2, 3, 1)
        self.ConvTranspose2d = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, output_padding=0)
    def forward(self, x1):
        v1 = self.Conv2d(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v20 = self.ConvTranspose2d(x1)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)

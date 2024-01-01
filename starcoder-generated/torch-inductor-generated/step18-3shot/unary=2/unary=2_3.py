
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.ConvTranspose2d(3, 5, kernel_size=(1, 1), stride=(1, 2), padding=(0, 1))
        self.t2 = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1, dilation=(2, 4))
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.t2(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)

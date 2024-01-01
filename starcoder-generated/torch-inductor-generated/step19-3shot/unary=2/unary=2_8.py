
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 5, 1, stride=2, padding=0, output_padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(2, 1, 1, stride=1, padding=0, output_padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(1, 7, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
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
x1 = torch.randn(2, 3, 6, 6)

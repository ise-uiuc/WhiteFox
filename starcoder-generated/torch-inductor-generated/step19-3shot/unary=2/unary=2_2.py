
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_l = torch.nn.Conv2d(1, 3, kernel_size=(5, 5), stride=(2, 2))
        self.conv_u = torch.nn.Conv2d(1, 9, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1))
        self.conv_r = torch.nn.Conv2d(9, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1))
    def forward(self, x1):
        v1 = self.conv_l(x1)
        v2 = self.conv_u(x1)
        v3 = self.conv_r(v2)
        v4 = v1 + v3 * 0.5
        v5 = v3 * v3 * v3
        v6 = v5 * 0.044715
        v7 = v3 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v4 * v10
        return v11
# Inputs to the model
x1 = torch.randn(2, 1, 10, 10)

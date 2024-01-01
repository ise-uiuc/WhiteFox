
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 1000)
        self.convt1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(26, 26), stride=(2, 2), padding=(0,), dilation=(1,))
        self.leakyrecti = torch.nn.LeakyReLU()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=[1000, 1, 1])
        self.linear2 = torch.nn.Linear(1000, 11968)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(0)
        v4 = self.convt1(v3)
        v5 = self.leakyrecti(v4)
        v6 = v5.squeeze(0)
        v7 = self.layernorm(v6)
        v8 = self.linear2(v7)
        y = torch.sigmoid(v8)
        return y
# Inputs to the model
x1 = torch.randn(1, 256, 1)

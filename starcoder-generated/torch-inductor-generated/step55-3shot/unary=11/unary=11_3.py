
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 7, 2, stride=1, padding=1)
        self.linear = torch.nn.Linear(3, 7, bias=False)
        self.softmax = torch.nn.Softmax(dim=0)
        self.t = torch.Tensor([[1, 2, 3], [-1000, 3, -100], [100, -1000, 1]])
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.linear(v5)
        v7 = self.t * v6
        v8 = self.softmax(v7 + v5)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)

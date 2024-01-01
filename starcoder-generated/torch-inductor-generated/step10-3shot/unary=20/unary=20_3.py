
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = torch.ops.aten.conv_transpose1d
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
        v1 = self.conv_trans(x1, x2, x3, stride=[2, 1], padding=[1, 1], output_padding=[1, 0])
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 186)
x2 = torch.randint(-2, 2, (1,), dtype=torch.int64)
x3 = torch.randint(-2, 2, (1,), dtype=torch.int64)
x4 = torch.randint(-2, 2, (1,), dtype=torch.int64)
x5 = torch.randint(-2, 2, (1,), dtype=torch.int64)
x6 = torch.randn(1, 1, 1)
x7 = torch.randn(1, 1, 1)
x8 = torch.randn(1, 1, 1)
x9 = torch.randn(1, 1, 1)
x10 = torch.randn(1, 1, 1)
x11 = torch.randn(1, 1, 1)
x12 = torch.randn(1, 1, 1, 1)

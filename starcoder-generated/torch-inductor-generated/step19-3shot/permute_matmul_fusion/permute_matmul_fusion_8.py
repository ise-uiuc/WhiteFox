
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute2 = torch.Tensor.permute
        self.add = torch.Tensor.__add__
    def forward(self, x1, x2):
        i1 = self.permute2(x1, 1, 2, 0)
        i2 = self.permute2(x2, 0, 2, 1)
        i3 = self.add(i1, i2)
        i4 = self.permute2(i3, 2, 1, 0)
        i5 = self.permute2(x2, 1, 2, 0)
        i6 = self.add(i4, i5)
        v1 = self.permute2(i6, 1, 2, 0)
        v2 = x1.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v4 = torch.matmul(v1, v2)
        v5 = torch.matmul(v3, v4)
        v6 = x1.permute(0, 2, 1)
        v7 = v5 + v6
        v8 = torch.matmul(v3, v7) # this invocation should trigger pattern
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

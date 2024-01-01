
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.functional.conv_transpose2d
        self.mul = torch.mul
        self.add = torch.add
        self.tanh = torch.tanh
    def forward(self, x1):
        v1 = self.conv_transpose(x1, torch.tensor([-1.74458144e+00,  3.73224828e-01], dtype=torch.float64), stride=[-1, 2], padding=[[0, 0], [1, 1]])
        v2 = self.mul(v1, torch.tensor([0.00000000e+00,  5.00000000e-01], dtype=torch.float64))
        v3 = self.mul(v1, v1)
        v4 = self.mul(v3, torch.tensor([0.00000000e+00,  2.23558630e-02], dtype=torch.float64))
        v5 = self.add(v1, v4)
        v6 = self.mul(v5, torch.tensor([0.00000000e+00,  7.97884561e-01], dtype=torch.float64))
        v7 = self.tanh(v6)
        v8 = self.add(v7, torch.tensor([1.00000000e+00,  0.00000000e+00], dtype=torch.float64))
        v9 = self.mul(v2, v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

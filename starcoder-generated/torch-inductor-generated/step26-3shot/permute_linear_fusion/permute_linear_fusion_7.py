
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.ops.aten.add
        self.expand = torch.ops.aten.expand
        self.linear = torch.nn.Linear(16, 16)
        self.mul = torch.ops.aten.mul
        self.permute = torch.ops.aten.permute
        self.relu6 = torch.ops.aten.relu6
        self.sub = torch.ops.aten.sub
    def forward(self, x1):
        v1 = self.permute(x1, (0, 2, 1))
        v2 = self.linear(self.sub(v1, 0.5), self.linear.weight, self.linear.bias)
        v3 = self.mul(v2, 0.25)
        v4 = self.linear(self.sub(v1, 0.5), self.linear.weight.transpose(-2, -1), self.linear.bias)
        v3 = self.sub(v3, 0.25)
        v4 = self.relu6(v4)
        v4 = self.sub(v3, v4)
        v5 = self.relu6(self.add(v4, 0.25))
        v6 = self.expand(v5, (v5.shape[0], v5.shape[2]))
        v5 = self.mul(v6, self.linear.weight.transpose(-2, -1))
        v5 = self.expand(v5, (x1.shape[0], x1.shape[2]))
        v5 = self.relu6(v5)
        v5 = self.add(v5, 0.25)
        v5 = self.mul(v5, v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 16)

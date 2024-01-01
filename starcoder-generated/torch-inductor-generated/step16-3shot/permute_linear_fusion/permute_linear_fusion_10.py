
class t1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
class t2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = t1()
        self.t2 = t2()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.t1.linear.weight, self.t1.linear.bias)
        v2 = torch.ops.prim.NumToTensor(v2)
        v3 = torch.nn.functional.relu(v2)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.ops.prim.NumToTensor(v1)
        v4 = torch.matmul(v2, v3)
        v5 = (torch.mul(v2, v3))
        v5 = torch.nn.functional.relu(v5)
        v5 = torch.add(v5, v3)
        v6 = torch.ops.prim.NumToTensor(v5)
        v5 = v5 * v4
        v6 = torch.nn.functional.relu(v6)
        v5 = v5 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)

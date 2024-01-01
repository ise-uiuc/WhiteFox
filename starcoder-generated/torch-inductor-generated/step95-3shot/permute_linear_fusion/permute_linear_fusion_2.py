
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.tanh(x1).permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        v3 = torch.nn.functional.tanh(v3)
        v4 = v3 * v1
        v4 = v4 + v1
        x2 = x2.permute(0, 2, 1)
        return x2.permute(0, 2, 1).matmul(v4)
def func(x):
    v1 = torch.nn.functional.tanh(x).permute(0, 2, 1)
    v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
    x2 = torch.nn.functional.relu(v2)
    v3 = x2.detach()
    v3 = torch.nn.functional.tanh(v3)
    v4 = v3 * v1
    v4 = v4 + v1
    x2 = x2.permute(0, 2, 1)
    return x2.permute(0, 2, 1).matmul(v4)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.func = func
    def forward(self, x1):
        return self.func(x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

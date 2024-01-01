
class Sub1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v0.permute(0, 2, 1)
        return v1

class Sub2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear2.weight, self.linear2.bias)
        v1 = v0.permute(0, 2, 1)
        return v1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = Sub1()
        self.sub2 = Sub2()
    def forward(self, x1):
        v0 = torch.nn.functional.linear(x1, self.sub1.linear1.weight, self.sub1.linear1.bias) + torch.nn.functional.linear(x1, self.sub2.linear2.weight, self.sub2.linear2.bias)
        v1 = v0.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 2, 2)

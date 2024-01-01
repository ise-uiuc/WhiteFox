
class Model_submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear_1.weight, self.linear_1.bias)
        v1_tmp = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(x2, self.linear_2.weight, self.linear_2.bias)
        a1 = torch.mm(v1_tmp, v2.permute(0, 2, 1))
        return a1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_module = Model_submodule()
    def forward(self, x1, x2):
        v1 = self.sub_module(x1, x2)
        return v1.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
# Inputs to the model

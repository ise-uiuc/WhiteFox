
# Module is not registered into a module_list.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        a1 = v2 + 1.0
        v3 = torch.nn.functional.linear(a1, self.linear2.weight, self.linear2.bias)
        v4 = v3.permute(0, 2, 1)
        a2 = v4.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(a2, self.linear3.weight, self.linear3.bias)
        return v5
# Inputs to the model
x1 = torch.randn(3, 2, 2)

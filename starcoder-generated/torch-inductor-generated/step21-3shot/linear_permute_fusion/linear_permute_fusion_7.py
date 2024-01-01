
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x4):
        t1 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        t2 = torch.nn.functional.linear(t1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(t2, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v1 = v2.permute(0, 2, 1)
        return v1
# Inputs to the model
x4 = torch.randn(2, 2, 2, 2)

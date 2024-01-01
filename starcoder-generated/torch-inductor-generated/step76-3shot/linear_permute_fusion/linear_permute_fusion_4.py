
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v4 = v0 + torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v4.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        return v3
# Inputs to the model
x0 = torch.randn(1, 2, 2)

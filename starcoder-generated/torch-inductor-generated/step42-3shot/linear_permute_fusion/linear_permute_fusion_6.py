
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = torch.nn.functional.linear(v0, self.linear2.weight, self.linear2.bias)
        v2 = v1.contiguous()
        return v2
# Inputs to the model
x0 = torch.randn(1, 2, 2)

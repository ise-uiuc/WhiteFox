
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        t1 = torch.nn.functional.pad(v1, (2, 2), value=0)
        v2 = torch.nn.functional.linear(t1, self.linear.weight, self.linear.bias)
        return torch.nn.functional.pad(v2, (2, 2), value=1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

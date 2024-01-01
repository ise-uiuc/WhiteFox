
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
        self.l1 = torch.nn.Linear(8, 8)
        self.l2 = torch.nn.Linear(8, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.l1.weight, self.l1.bias)
        v2 = torch.nn.functional.linear(v1, self.l2.weight, self.l2.bias)
        return v2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(5, 2, 2)

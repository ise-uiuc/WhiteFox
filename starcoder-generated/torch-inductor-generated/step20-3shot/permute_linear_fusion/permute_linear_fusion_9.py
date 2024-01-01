
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        # Note that this statement has no input tensors yet.
        self.linear1 = torch.nn.Linear(2, 2)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.linear1(self.linear.weight.permute(0, 2, 1))
        return v1 + v3
x1 = torch.randn(1, 2, 2)

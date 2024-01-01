
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        if (x1 == 5).all():
            v1 = x1.permute(2, 1, 0)
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            v3 = torch.nn.functional.relu(v2)
        else:
            v4 = x1.permute(0, 2, 1)
            v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
            x2 = torch.nn.functional.relu(v5)
            v6 = x2.detach()
            v3 = x2.permute(0, 2, 1)
        return torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

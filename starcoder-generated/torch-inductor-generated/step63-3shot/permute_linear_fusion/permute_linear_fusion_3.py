
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.flatten(v1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)

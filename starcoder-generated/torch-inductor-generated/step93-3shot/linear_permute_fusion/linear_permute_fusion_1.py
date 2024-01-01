
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = torch.nn.functional.linear(torch.transpose(v3, 1, 2), self.linear.weight, self.linear.bias)
        v5 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v6 = torch.nn.functional.linear(torch.transpose(v5, 0, 1), self.linear.weight, self.linear.bias)
        return v4 + v6
# Inputs to the model
x = torch.randn(1, 2, 2)

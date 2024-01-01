
class Model(torch.nn.Module):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.linear = torch.nn.Linear(b, d)
    def forward(self, input):
        v1 = torch.nn.functional.linear(input, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        return v1 + v2 + v3 + v4 + v5
# Inputs to the model
input = torch.randn(1, 2, 2)

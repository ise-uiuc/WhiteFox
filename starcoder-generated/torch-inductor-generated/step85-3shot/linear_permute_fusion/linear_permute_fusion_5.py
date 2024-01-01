
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x2):
        v4 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias) # noqa E0633
        v2 = v4.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v4
# Inputs to the model
x2 = torch.randn(3, 1, 2)

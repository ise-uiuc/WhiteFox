
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.squeeze()
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v2.unsqueeze()
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

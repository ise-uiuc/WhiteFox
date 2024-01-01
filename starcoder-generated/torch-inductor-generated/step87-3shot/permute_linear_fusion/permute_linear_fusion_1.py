
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = self.sigmoid(v1)
        v3 = x2.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v3 = v3.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        x3 = x2 + v4
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 3)

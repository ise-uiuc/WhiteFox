
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu6 = torch.nn.ReLU6(inplace=False)
    def forward(self, x1):
        x2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = self.relu6(x2)
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = self.relu6(v2)
        return v2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

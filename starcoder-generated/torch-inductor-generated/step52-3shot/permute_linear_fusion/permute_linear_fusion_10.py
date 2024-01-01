
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x1 = x1.permute(0, 2, 1)
        x2 = torch.nn.functional.softmax(x1, dim=1)
        return torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

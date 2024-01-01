
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = self.relu(v2)
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        x4 = self.relu(v2)
        x3 = torch.max(v3, x4)
        v2 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 3)

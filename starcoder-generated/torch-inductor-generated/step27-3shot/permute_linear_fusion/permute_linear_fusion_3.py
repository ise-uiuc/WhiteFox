
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.permute(0, 2, 1)
        x2 = torch.nn.functional.relu(v3)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v3)
        v4 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)

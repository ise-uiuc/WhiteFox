
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x3 = torch.relu(v1)
        x1 = torch.randn(1, 3, 1) / 2
        v2 = torch.nn.functional.linear(x1, self.linear.weight)
        x3 = torch.sigmoid(v2) * torch.nn.functional.linear(x1, self.linear.bias / 2)
        v3 = torch.cat(v1, x3)
        v1 = v3 + v3
        x2 = torch.nn.functional.linear(x1, self.linear.weight)
        x2 = v1.view(v1.size())
        x2 = v3 + x2
        v3 = torch.nn.functional.linear(x2, self.linear.bias + x3)
        return torch.nn.functional.linear(v3 / 1, self.linear.weight)
# Inputs to the model
x1 = torch.randn(1, 1, 1)

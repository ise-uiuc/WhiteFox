
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = torch.nn.functional.relu(v1).permute(0, 3, 2, 1)
        v3 = torch.clamp(v2, min=-2, max=2)
        return torch.nn.functional.sigmoid(torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias))
# Inputs to the model
x1 = torch.randn(2, 10, 10, 10)

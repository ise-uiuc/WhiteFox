
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight)
        x2 = self.ReLU(v2)
        y = torch.relu(self.linear.bias)
        return (y + x2) * x1
# Inputs to the model
x1 = torch.randn(2, 2, 2)

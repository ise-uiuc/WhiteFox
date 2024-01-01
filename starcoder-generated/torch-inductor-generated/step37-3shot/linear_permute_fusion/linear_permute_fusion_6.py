
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.relu(v2)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)

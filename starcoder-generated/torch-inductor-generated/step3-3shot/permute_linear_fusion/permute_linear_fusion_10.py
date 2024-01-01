
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.relu(x1.permute(0, 2, 1))
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)

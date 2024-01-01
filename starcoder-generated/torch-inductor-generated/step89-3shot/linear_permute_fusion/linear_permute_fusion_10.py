
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 3, True)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1 + v1
        v3 = self.relu(v2)
        v4 = v3 * v3
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = self.relu(v1)
        v3 = self.linear2(v2)
        return v1
# Inputs to the model
x3 = torch.randn(2, 2, 2, 2)

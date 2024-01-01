
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        y = self.flatten(x1)
        return torch.nn.functional.relu(torch.nn.functional.linear(y, self.linear1.weight, self.linear1.bias))
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

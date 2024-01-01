
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v2 = torch.nn.functional.linear(x1.detach(), self.linear.weight, self.linear.bias)
        return self.relu(v2)
# Inputs to the model
x1 = torch.randn(2, 2, 1)

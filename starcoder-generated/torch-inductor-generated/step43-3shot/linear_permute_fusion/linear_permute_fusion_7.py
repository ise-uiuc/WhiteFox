
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v1 = torch.nn.functional.relu(x0)
        v0 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v0.softmax(dim=1)
        return v2
# Inputs to the model
x0 = torch.randn(3, 2)

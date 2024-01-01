
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 12)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1)
        v2 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v3 = v1.view(1, 10, 10, 1)
        v4 = torch.nn.functional.interpolate(v3, scale_factor=2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 10, 10)

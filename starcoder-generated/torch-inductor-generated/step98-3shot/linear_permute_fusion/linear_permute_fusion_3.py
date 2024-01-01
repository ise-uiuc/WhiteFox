
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.reshape = torch.nn.Flatten()
    def forward(self, x):
        v0 = x
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = self.reshape(v1)
        return v0
# Inputs to the model
x = torch.randn(1, 2, 2, 2)

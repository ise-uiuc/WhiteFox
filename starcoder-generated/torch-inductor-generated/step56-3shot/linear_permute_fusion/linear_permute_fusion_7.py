
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.pool = torch.nn.MaxPool2d(2, 1, 1)
    def forward(self, x1):
        v9 = x1
        v1 = torch.nn.functional.linear(v9, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = self.pool(v2)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
print(x1.shape)

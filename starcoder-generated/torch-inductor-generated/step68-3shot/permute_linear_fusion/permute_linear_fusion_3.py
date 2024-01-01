
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.reshape(-1, 2)
        v4 = v3
        v4 = self.relu(v4)
        v4 = v4.reshape(x1.shape)
        v4 = v2 - v4
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)

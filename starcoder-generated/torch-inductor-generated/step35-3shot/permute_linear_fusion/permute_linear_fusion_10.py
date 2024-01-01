
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.reshape(x2, (1, 4))
        v4 = torch.nn.functional.avg_pool1d(v3, padding=1)
        v5 = v4.permute(0, 2, 1)
        v6 = torch.nn.functional.relu(v5)
        v7 = torch.reshape(v6, (1, 2))
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dropout = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = self.dropout(v2)
        x3 = torch.nn.functional.normalize(x2)
        y = torch.matmul(x3, self.linear.bias)
        z = torch.nn.functional.relu(y)
        return z
# Inputs to the model
x1 = torch.randn(1, 1, 1)

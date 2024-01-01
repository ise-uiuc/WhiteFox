
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        x2 = self.dropout(v4)
        y = torch.matmul(x2, self.linear.bias)
        z = torch.nn.functional.relu(y)
        return z
# Inputs to the model
x1 = torch.randn(1, 2, 2)

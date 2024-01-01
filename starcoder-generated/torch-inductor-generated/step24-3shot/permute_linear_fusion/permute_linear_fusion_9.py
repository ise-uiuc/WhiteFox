
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = v3.transpose(1, 0)
        x2 = torch.nn.functional.relu(v4)
        return x2.transpose(1, 0)
# Inputs to the model
x1 = torch.randn(2, 2, 2)

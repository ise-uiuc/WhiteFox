
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.tanh(v2)
        v4 = v3.permute(0, 2, 1)
        v5 = torch.sigmoid(v4)
        v6 = v5.squeeze()
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)

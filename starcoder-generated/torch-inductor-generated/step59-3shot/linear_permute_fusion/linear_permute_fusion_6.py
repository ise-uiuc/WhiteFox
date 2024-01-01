
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.sigmoid
    def forward(self, x1):
        v2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        return v3.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

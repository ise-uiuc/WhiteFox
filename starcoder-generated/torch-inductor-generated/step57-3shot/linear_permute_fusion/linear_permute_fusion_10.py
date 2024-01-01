
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        # Here is inserted another linear operation
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)

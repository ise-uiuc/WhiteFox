
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(torch.reshape(x1, (1, 2, 2)), torch.reshape(self.linear.weight, (2, 2)), torch.reshape(self.linear.bias, (2)))
        v2 = v1.permute(0, 2, 1)
        return torch.reshape(v2, (2, 2, 2))
# Inputs to the model
x1 = torch.randn(1, 2, 2)

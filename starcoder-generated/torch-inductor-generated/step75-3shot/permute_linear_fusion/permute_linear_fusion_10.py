
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.view = torch.reshape
        self.bilinear = torch.nn.Bilinear(2, 2, 1)
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.view(v2, (1, 4))
        return torch.nn.functional.bilinear(v1, x2, self.bilinear.weight, self.bilinear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

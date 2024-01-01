
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1, x2, x3):
        x4 = x1.permute(0, 2, 1)
        x5 = x2.permute(0, 2, 1)
        x6 = x3.permute(0, 2, 1)
        x7 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        x7 = x7.permute(0, 2, 1)
        x8 = torch.nn.functional.linear(x6, self.linear.weight, self.linear.bias)
        x8 = x8.permute(0, 2, 1)
        x8 = x7 + x8
        x9 = torch.matmul(x5, x8)
        return x9
# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(1, 3, 3)
x3 = torch.randn(1, 3, 3)

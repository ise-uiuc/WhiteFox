
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
        self.linear3 = torch.nn.Linear(2, 2)
        self.linear4 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        x3 = torch.abs(x2)
        x4 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)
        x5 = torch.nn.functional.elu(x4)
        x1 = x2 + x5.to(x2.dtype)
        x4 = x4 + self.linear3.weight
        x4 = (-1) * x4.clamp(min=0) - (-1) * x4.clamp(max=0)
        x1 = x1.permute(0, 2, 1)
        x4 = torch.nn.functional.linear(x4, self.linear4.weight, self.linear4.bias)
        return x1
# Inputs to the model
x1 = torch.randn(1,2, 2)

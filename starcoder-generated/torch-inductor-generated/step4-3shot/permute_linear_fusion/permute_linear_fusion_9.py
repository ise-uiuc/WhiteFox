
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        t1 = x1.permute(0, 1, 2)
        t2 = torch.nn.functional.linear(t1, self.linear.weight, self.linear.bias)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 2)

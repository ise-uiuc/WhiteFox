
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.linear1 = torch.nn.Linear(4, 4)
    def forward(self, x):
        x1 = x.permute(0, 2, 1)
        x1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x1 = x1 * -10.51
        x1 = self.linear1(x1)
        x1 = x1.permute(0, 2, 1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 12)
    def forward(self, x0, *size):
        v0 = x0
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        shape = [-1] + list(size)
        return v2.view(shape)
# Inputs to the model
x0 = torch.randn(1, 2, 2)
size = 2, 3

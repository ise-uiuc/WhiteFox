
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x[0], self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v1
        v4 = v2
        return (v2, v4)
# Inputs to the model
x = torch.Tensor([[]]), torch.randn(2, 3, 2)

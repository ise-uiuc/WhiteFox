
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x_0):
        v1 = torch.nn.functional.linear(input_1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2.flatten(0, 1) + 2
# Inputs to the model
x_0 = torch.randn(1, 2, 2)

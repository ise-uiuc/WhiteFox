
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        y = torch.nn.functional.linear(input.reshape(-1, 2), self.linear.weight, self.linear.bias).reshape(3, 2, 2)
        return y
# Inputs to the model
input = torch.randn(1, 3, 2, 2)

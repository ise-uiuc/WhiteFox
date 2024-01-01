
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, input):
        v0 = input
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = (v1 - v0).permute(0, 2, 1)
        return v2 + v0
# Inputs to the model
input = torch.ones((1, 2, 2)),

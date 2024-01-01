
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(200, 20)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(1, 0, 2)
        return v3
# Inputs to the model
x1 = torch.randn(200, 1, 2)

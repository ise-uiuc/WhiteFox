
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(8, 8)
    def forward(self, x1):
        v1 = torch.nn.functional.tanh(x1).permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 8, 8)

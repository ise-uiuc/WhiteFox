
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = torch.unsqueeze(x1, 0)
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        a1 = v1.permute(0, 2, 1)
        a2 = torch.nn.functional.relu(a1)
        return (a1, a2)
# Inputs to the model
x1 = torch.randn(3, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.flip(-1)
        v2 = torch.nn.functional.relu(self.linear(v1))
        return torch.nn.functional.relu(self.linear(v2))
# Inputs to the model
x1 = torch.randn(1, 2, 2, requires_grad=False)

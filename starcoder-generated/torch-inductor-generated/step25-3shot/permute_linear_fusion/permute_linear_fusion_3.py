
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v1 = self.linear(x)
        x = x * v1.detach()
        x = torch.nn.functional.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)

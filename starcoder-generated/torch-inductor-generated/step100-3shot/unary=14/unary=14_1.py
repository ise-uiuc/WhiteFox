
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear_1 = torch.nn.Linear(45, 9)
    def forward(self, x1):
        v1 = self.flatten(x1)
        v2 = self.linear_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

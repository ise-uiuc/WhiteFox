
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.permute(1, 0, 2)
        return v2.flatten(0, 1)
# Inputs to the model
x1 = torch.randn(2, 2, 2)

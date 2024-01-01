
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v0 = x0.permute(0, 2, 1)
        v1 = v0.squeeze(0)
        v2 = self.linear(v1)
        return v2
# Inputs to the model
x0 = torch.randn(2, 2, 3)

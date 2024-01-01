
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v2 = x1 + x2
        v3 = v2.detach()
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)

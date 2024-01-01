
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = self.linear(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)

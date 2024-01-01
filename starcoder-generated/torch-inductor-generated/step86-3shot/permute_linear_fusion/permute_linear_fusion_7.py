
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.flatten(v1)
        v3 = v2.permute(1, 0, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2)

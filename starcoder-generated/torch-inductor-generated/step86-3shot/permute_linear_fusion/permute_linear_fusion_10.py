
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1.squeeze(1)
        return self.flatten(v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)

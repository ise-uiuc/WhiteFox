
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 1, stride=1, padding=1),
            torch.nn.Conv2d(8, 16, 1, stride=1, padding=1))
    def forward(self, x1, other=False):
        v1 = self.model(x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)

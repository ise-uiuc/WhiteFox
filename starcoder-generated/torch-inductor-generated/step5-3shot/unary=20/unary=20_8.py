
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, stride=1, padding=0), torch.nn.Conv2d(2, 4, 5, stride=1, padding=0), torch.nn.Conv2d(4, 8, 2, stride=1, padding=0), torch.nn.Sigmoid())
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)

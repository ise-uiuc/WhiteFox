
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Sequential(torch.nn.Conv2d(3,  32, 3, stride=2, padding=1), torch.nn.ReLU())
        self.features2 = torch.nn.Sequential(torch.nn.GroupNorm(32, 32), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.ReLU())
        self.features3 = torch.nn.Sequential(torch.nn.GroupNorm(32, 64), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.features1(x1)
        v2 = self.features2(v1)
        v3 = self.features3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

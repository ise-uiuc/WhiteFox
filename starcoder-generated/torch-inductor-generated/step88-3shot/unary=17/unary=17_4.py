
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 32, 1, stride=2), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 1))
        self.module_1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 64, 2, stride=2), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 1))
    def forward(self, x1):
        v1 = self.module_0(x1)
        v2 = self.module_1(x1)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(256, 256, [4, 3], stride=[2, 2], padding=[0, 1]), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 256, 1792, 1)

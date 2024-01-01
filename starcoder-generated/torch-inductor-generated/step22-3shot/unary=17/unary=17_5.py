
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 3, [3, 4], stride=[2, 1], padding=(0, 1)), torch.nn.Sigmoid())
        self.module_1 = torch.nn.Sequential(torch.nn.Conv2d(8, 1, [7, 5], stride=[2, 1], padding=(1, 3)), torch.nn.Sigmoid())
        self.module_2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 8, [4, 5], stride=[2, 1], padding=[1, 2]), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.module_0(x1)
        v2 = self.module_1
        v3 = self.module_2(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 93, 20)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
        self.module_1 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
        self.module_2 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
        self.module_3 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
        self.module_4 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
        self.module_5 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ConvTranspose2d(3, 3, [3, 4], stride=[1, 2], padding=(1, 3)), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.module_0(x1)
        v2 = self.module_1(v1)
        v3 = self.module_2(v2)
        v4 = self.module_3(v3)
        v5 = self.module_4(v4)
        v6 = self.module_5(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

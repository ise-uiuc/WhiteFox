
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True),
            torch.nn.ReLU())
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            torch.nn.ReLU())
    def forward(self, z):
        v1 = self.layer_1(z)
        v2 = self.layer_2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 49, 512, 437)
x1 = x[:, 0:1,...]
x2 = x[:, 2:3,...]

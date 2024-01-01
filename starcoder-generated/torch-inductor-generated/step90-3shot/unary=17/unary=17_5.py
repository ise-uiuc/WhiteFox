
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ConvTranspose3d(3, 8, (5, 5, 5), (2, 2, 2), (3, 3, 3), (1, 1, 1), bias=False), torch.nn.ReLU(), torch.nn.Sigmoid())
    def forward(self, x):
        x1 = self.block(x)
        return x1
# Inputs to the model
x = torch.randn(1, 3, 16, 16, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 8, [3, 4], stride=[1, 2], padding=(1, 3)),
            torch.nn.ReLU(),
        )
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 96, 9)

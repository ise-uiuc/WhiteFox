
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.downconvs = torch.nn.Sequential(*([torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)]*2))
    def forward(self, x1):
        v1 = self.downconvs(x1)
        v2 = torch.nn.Sigmoid()(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

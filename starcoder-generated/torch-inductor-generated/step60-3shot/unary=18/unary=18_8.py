
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.Sequential( torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(4), torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(12), torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(1), torch.nn.Sigmoid() )
    def forward(self, x1):
        v1 = self.convs(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)

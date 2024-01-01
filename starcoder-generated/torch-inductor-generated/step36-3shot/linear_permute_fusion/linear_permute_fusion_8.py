
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2), stride=(2, 2), bias=None)
# Inputs to the model
        weight = torch.randn(2, 1, 2, 2, device='cuda')
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2), stride=(2, 2), bias=None).cuda().weight.data = weight
    def forward(self, x1):
        v0 = self.conv(x1)
        return v0


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.add = torch.add
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.add(v1, torch.tensor([[0., 1.], [2., 3.], [4., 5.]]))
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

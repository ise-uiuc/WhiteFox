
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=64, groups=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x0):
        v0 = self.conv2d(x0)
        v1 = torch.sum(v0, dim=tuple(range(1, v0.dim()))) # Sum all elements along dimensions starting at dimension 1, since dimension 0 is the batch dimension. We cannot sum the batch dimension, but the values in dimension 1 represent channels.
        return v1
# Inputs to the model
x0 = torch.randn(1, 3, 32, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)
    def forward(self, x1):
        # TODO: insert input tensor dimensions such that output of conv matches input tensor batch size
        v1 = self.conv(x1)
        v2 = v1.permute(0, 2, 3, 1)
        v2.to(torch.float32)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)

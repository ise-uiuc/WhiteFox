
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 56, 5, stride=2, padding=1)
    def forward(self, x1, padding2=None, other=None):
        v1 = self.conv(x1)
        if padding2 == None:
            padding2 = torch.randint(0, 16, size=[], dtype=torch.int64)
        v2 = v1 + torch.randn(v1.shape)
        v3 = torch.flatten(v2, 1)
        v4 = v3 + torch.randn(v3.shape).to(x1.device)
        v5 = torch.nn.functional.interpolate(v4, scale_factor=1.5, mode='bicubic')
        v6 = torch.nn.functional.conv2d(v5, torch.ones([4, 4, 16, 4]))
        return v6
# Inputs to the model
x1 = torch.randn(1, 23, 56, 56).to('cpu')

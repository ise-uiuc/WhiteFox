
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(65536, 4096, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        # PyTorch is not yet able to resolve x1 and self.conv
        x1 = torch.randn(2785, 65536, 7, 7)
        x2 = self.conv_t(x)
        x3 = x2 > 0
        x4 = x2 * 0.3029
        x5 = torch.where(x3, x2, x4)
        x6 = torch.cat([x5, x1], dim=0)
        x7 = torch.mean(x6, dim=0)
        x8 = torch.prod(x7, dim=3, keepdim=True)
        return torch.nn.functional.adaptive_max_pool2d(x8, [1, 1])
# Inputs to the model
x = torch.randn(256, 2785, 7, 7)

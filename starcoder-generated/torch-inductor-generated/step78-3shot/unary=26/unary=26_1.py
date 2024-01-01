
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 128, kernel_size=(7, 5), stride=(5, 2), padding=(0, 2), groups=2, bias=True)
    def forward(self, input):
        out = self.conv_t(input)
        mask = out > 0
        mul = out * -0.290
        out = torch.where(mask, out, mul)
        mask = out > 0
        mul = out * -0.743
        out = torch.where(mask, out, mul)
        out = torch.nn.functional.interpolate(out, size=(256, 2), scale_factor=None)
        return torch.nn.functional.adaptive_max_pool2d(out, (1, 1))
# Inputs to the model
input = torch.randn(12, 128, 156, 155)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(56, 76, 7, stride=3, padding=1, output_padding=5, bias=False, dilation=1)
    def forward(self, hidden):
        out = self.conv_t(hidden)
        mask = out > 0
        mul = out * 0.03
        out = torch.where(mask, out, mul)
        out = torch.nn.functional.adaptive_max_pool2d(out, (55, 139))
        out = torch.nn.functional.layer_norm(out, normalized_shape=[1, 1, 82, 171], weight=None, bias=None, eps=1e-05)
        return torch.nn.functional.layer_norm(out, normalized_shape=[1, 1, 82, 171], weight=None, bias=None, eps=1e-05)
# Inputs to the model
hidden = torch.randn(24, 56, 70, 104)

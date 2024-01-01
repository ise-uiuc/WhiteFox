
class Model(torch.nn.Module):
    def __init__(self, min_value=1.3862944, max_value=1.4142135):
        super().__init__()
        self.conv_transpose = torch.nn.ModuleList(torch.nn.ModuleList(torch.nn.ConvTranspose2d(1, 8, 4, stride=1, padding=2, output_padding=0)))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose[0][0](x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

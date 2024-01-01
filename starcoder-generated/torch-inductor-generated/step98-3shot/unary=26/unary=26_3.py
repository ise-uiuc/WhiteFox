
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(77, 37, 1, stride=1, padding=0, bias=False)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * -0.773
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.Sigmoid()(v4), (1, 1))
# Inputs to the model
x2 = torch.randn(19, 77, 45, 3)

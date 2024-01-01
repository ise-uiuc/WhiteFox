
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(5, 8, 5, padding=3, stride=2)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 > 0
        y3 = y1 * 0.961
        y4 = torch.where(y2, y1, y3)
        return y4 + torch.nn.functional.adaptive_avg_pool1d(y4, 1)
# Inputs to the model
x = torch.randn(13, 5, 41)

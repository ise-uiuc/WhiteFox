
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(94, 94, 1, stride=2, padding=0, bias=False)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 > 4.218
        y3 = y1 * -0.599
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.relu(y4)
# Inputs to the model
x = torch.randn(1, 94, 6, 93)

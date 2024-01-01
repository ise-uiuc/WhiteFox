
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, 3, stride=1, padding=1, output_padding=1)
    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

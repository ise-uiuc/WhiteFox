
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
    def forward(self, img):
        img = self.conv(img)
        v2 = img.permute(0, 2, 3, 1)
        res = torch.sum(v2, axis=-1)
        return res
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

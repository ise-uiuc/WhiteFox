
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv transpose 1 = torch.nn.ConvTranspose2d(75, 191, 14, stride=2, padding=1)
        self.conv transpose 1 = torch.nn.ConvTranspose2d(89, 151, 2, stride=1, padding=0, output_padding=0)
        self.conv transpose 2 = torch.nn.ConvTranspose2d(74, 33, 2, stride=1, padding=1, output_padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv transpose 1(x1)
        v1 = torch.sigmoid(v1)
        v2 = self.conv transpose 2(x2)
        v1 = v1 + v2
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 75, 200, 400)
x2 = torch.randn(1, 89, 435, 527)
x3 = torch.randn(1, 74, 870, 1053)

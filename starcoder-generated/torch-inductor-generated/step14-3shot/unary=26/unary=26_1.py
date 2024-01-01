
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_3 = torch.nn.ConvTranspose2d(64, 128, 1, stride=2, padding=1, dilation=1, output_padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        y = self.conv_t_3(x)
        y1 = y > 0
        y2 = y * -1
        y3 = torch.where(y1, y, y2)
        y4 = self.relu(y3)
        return y4
# Inputs to the model
x = torch.randn(1, 64, 21, 22)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = torch.softmax(self.conv_transpose(x1), dim=1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)

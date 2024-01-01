
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(50, 23, 4, stride=3)
        self.batch_norm = torch.nn.BatchNorm2d(86020)
    def forward(self, input):
        x = self.conv_transpose(input)
        x = self.batch_norm(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 50, 187, 1024)

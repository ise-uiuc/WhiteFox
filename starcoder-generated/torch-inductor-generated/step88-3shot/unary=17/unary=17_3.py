
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_transpose(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 64, 1, 1)

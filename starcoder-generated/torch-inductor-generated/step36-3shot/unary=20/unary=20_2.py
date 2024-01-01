
class Model(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=7, kernel_size=(1, 14), stride=(1, 7), padding='same')
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=7, out_channels=in_channels, kernel_size=(1, 14), stride=(1, 7), padding='same')
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = torch.relu(a1)
        a3 = self.conv_transpose(a2)
        a4 = torch.relu(a3)
        return a1 - a4
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)

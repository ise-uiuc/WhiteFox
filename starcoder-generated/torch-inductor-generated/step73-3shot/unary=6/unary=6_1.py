
class Model(nn.Module):
    def __init__(self):
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=2)

    def forward(self, x):
        conv = torch.flatten(x[:, :, :, :].unsqueeze(2), 2)
        conv = self.conv(conv)
        conv = torch.reshape(conv, (-1, 800))
        return conv
# Inputs to the model
x2 = torch.randn(2, 3, 64, 64)
x1 = [torch.randn(2, 1, 64, 64)]

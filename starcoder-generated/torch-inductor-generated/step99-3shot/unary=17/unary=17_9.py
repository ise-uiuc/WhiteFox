
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(1, 64, 3, stride=2, bias=False,
                                               padding=0, output_padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(64, 1, 3, stride=2, bias=False,
                                               padding=0, output_padding=0)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        return torch.sigmoid(self.conv1(torch.relu(self.conv0(x))))
# Inputs to the model
x1 = torch.randn(1, 1, 380, 540)

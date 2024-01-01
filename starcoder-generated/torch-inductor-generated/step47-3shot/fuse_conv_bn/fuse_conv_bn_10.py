
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 2)
        self.conv1 = torch.nn.Conv2d(4, 4, 2)
        self.conv2 = torch.nn.ConvTranspose2d(4, 4, 2)
        self.fc = torch.nn.Linear(4, 1)
    def forward(self, x):
        x = self.conv(x)
        z = self.conv1(x)
        z = torch.mean(z, dim=1, keepdim=True)
        x = torch.transpose(x, 1, 2)
        x = self.conv2(x, output_size=(z, x.shape[2]))
        x = torch.transpose(x, 1, 2)
        return x
# Inputs to the model
x = torch.randn(1, 1, 8, 8)

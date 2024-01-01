
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 64, kernel_size=(7, 7), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(64, 96, kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = torch.nn.ConvTranspose2d(96, 128, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.ConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    def forward(self, x1):
        out = torch.nn.functional.relu(self.conv1(x1))
        out = torch.add(-torch.nn.functional.relu(self.conv2(out)), out)
        out = torch.add(-torch.nn.functional.relu(self.conv3(out)), out)
        out = torch.add(-torch.nn.functional.relu(self.conv4(out)), out)
        return out
# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_1a = torch.nn.ConvTranspose2d(1, 128, 1, stride=1)
        self.conv2d_2a = torch.nn.ConvTranspose2d(128, 256, 1, stride=1)

        self.conv2d_1d = torch.nn.ConvTranspose2d(128, 64, 1, stride=1)
        self.conv2d_2d = torch.nn.ConvTranspose2d(256, 256, 1, stride=1)
        self.conv2d_3d = torch.nn.ConvTranspose2d(256, 256, 1, stride=1)

        self.conv2d_1s = torch.nn.ConvTranspose2d(64, 1, 3, stride=1)
        self.conv2d_2s = torch.nn.ConvTranspose2d(256, 128, 3, stride=1)
        self.conv2d_3s = torch.nn.ConvTranspose2d(256, 1, 2, stride=1)

    def forward(self, input):

        # Downsampling block:
        x1 = self.conv2d_1a(input)
        x1 = self.conv2d_2a(x1)
        x1 = torch.unsqueeze(input,dim=1)
        x1 = torch.cat((x1,x1),dim=1)
        x2 = self.conv2d_1d(x1)
        x2 = self.conv2d_2d(x2)
        x2 = self.conv2d_3d(x2)

        x3 = self.conv2d_1s(x2)
        x3 = self.conv2d_2s(x3)
        x3 = self.conv2d_3s(x3)

        return x3

# Inputs to the model
input = torch.randn(1, 1, 256, 256)

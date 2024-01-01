
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, in_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.conv2= torch.nn.Conv2d(3, 16, (3, 3), padding=1)
        self.conv3= torch.nn.Conv2d(16, out_channels=32, kernel_size=(1, 1), stride=(2, 2))
        self.conv4 = torch.nn.Conv2d(32, 64, (1, 1))
        self.conv5 = torch.nn.Conv2d(64, 100, (3, 3), padding=1)
        self.conv6 = torch.nn.Conv2d(100, 2, kernel_size=(1, 1), stride=(2, 2))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(x)
        x = v2 + v3
        v4 = self.conv3(x)
        v5 = torch.sigmoid(v4)
        v6 = self.conv4(v5)
        v7 = torch.relu(v6)
        v8 = self.conv5(v7)
        v9 = torch.relu(v8)
        v10 = self.conv6(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x = torch.randn(1, 3, 50, 50)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=96, out_channels=16, kernel_size=(8, 8), stride=(2, 2), padding=1, output_padding=(0,1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(4, 4), stride=(2, 2))

    def forward(self, x, y):
        v1 = self.conv_transpose1(x)
        v2 = torch.sigmoid(v1)
        v3 = torch.cat((y,v2),1)
        v4 = self.conv_transpose2(v3)
        v5 = torch.sigmoid(v4)
        return v5

x = torch.randn(1, 96, 16, 100)
y = torch.randn(1, 16, 8,101)
# model ends
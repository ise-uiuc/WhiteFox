
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= torch.nn.Conv2d(512, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2= torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3= torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4= torch.nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5= torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6= torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v2 = torch.sigmoid(v3)
        v2 = self.conv3(v2)
        v3 = torch.sigmoid(v2)
        v4 = self.conv4(v3)
        v3 = torch.sigmoid(v4)
        v5 = self.conv5(v3)
        v3 = torch.sigmoid(v5)
        v6 = self.conv6(v3)
        v3 = torch.sigmoid(v6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 17, 17)

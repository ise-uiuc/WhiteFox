
class conv_layer(torch.nn.Module):
    def __init__(self, c1, m):
        super(conv_layer, self).__init__()
        self.convtranspose1 = torch.nn.ConvTranspose2d(in_channels=c1, out_channels=m,
                                                       kernel_size=(1, 1), stride=(1, 1),
                                                       padding=(0, 0), bias=False)

    def forward(self, x):
        t1 = self.convtranspose1(x)
        t2 = torch.relu(t1)
        return t2


class FC_layer(torch.nn.Module):
    def __init__(self, c1, c2=6, b=1):
        super(FC_layer, self).__init__()
        self.convtranspose1 = conv_layer(c1, c2)
        self.bias = torch.empty(size=(c2), dtype=torch.float32)
        torch.nn.init.constant_(self.bias, -b)
        self.conv1d = torch.nn.Conv1d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.convtranspose2 = conv_layer(c2, c2)
        # self.conv2d = torch.nn.Conv2d(in_channels=c2, out_channels=c2,
        #                               kernel_size=(1, 1), stride=(1, 1),
        #                               padding=(0, 0), bias=False)

    def forward(self, x):
        y = self.convtranspose1(x)
        y = y + self.bias
        y = self.conv1d(y.view(y.shape[0], y.shape[1], -1))
        y = y + self.bias
        y = self.convtranspose2(y.view(y.shape[0], y.shape[1], y.shape[2], 1, y.shape[3]))
        # y = self.convtranspose2(y)
        return y


class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=4, out_channels=6, bias=False, padding=(0, 0), stride=(2, 2), kernel_size=(1, 1))
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=1, bias=False, kernel_size=(2, 2),
                                              stride=(1, 1), padding=(0, 0))
        self.fc1 = FC_layer(1, c2=16, b=0.3)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, bias=False, kernel_size=1,
                                              stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.fc1(x2)
        x4 = self.conv3(x3)
        # x4 = self.relu(x4)
        # x6 = self.conv6(x5)
        # x10 = x4 + x6
        return x4
#Inputs to the model
x1 = torch.randn(1, 4, 44, 44)

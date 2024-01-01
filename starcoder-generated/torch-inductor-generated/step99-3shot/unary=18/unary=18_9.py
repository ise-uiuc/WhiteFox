
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2), padding=2, dilation=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2), dilation=2)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2), padding=(2, 2), dilation=(2, 2))
        self.conv7 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2), dilation=(2, 2))
        self.conv8 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2, 2), stride=(2, 2), dilation=(2, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        n1 = torch.tanh(v2)
        n2 = torch.tanh(n1)
        v3 = self.conv2(n2)
        v4 = torch.sigmoid(v3)
        n3 = v4
        n4 = torch.tanh(n3)
        n5 = v1
        n6 = (n5 + n6)
        n7 = torch.tanh(n6)
        v5 = self.conv3(n7)
        v6 = torch.sigmoid(v5)
        n8 = torch.tanh(v6)
        n9 = torch.tanh(n8)
        v7 = self.conv4(n9)
        v8 = torch.sigmoid(v7)
        n10 = v8
        n11 = torch.tanh(n10)
        n12 = n3
        n13 = (n12 + n13)
        n14 = torch.tanh(n13)
        n15 = n3
        n16 = (n15 - n16)
        n17 = torch.tanh(n16)
        v9 = self.conv5(n14)
        v10 = torch.sigmoid(v9)
        n18 = torch.tanh(v10)
        n19 = torch.tanh(n18)
        v11 = self.conv6(n19)
        v12 = torch.sigmoid(v11)
        n20 = v12
        n21 = torch.tanh(n20)
        n22 = torch.tanh(n19)
        n23 = (n21 + n22)
        n24 = torch.tanh(n23)
        v13 = self.conv7(n24)
        v14 = torch.sigmoid(v13)
        n25 = v14
        n26 = torch.tanh(n25)
        n27 = torch.tanh(n24)
        n28 = (n26 + n27)
        n29 = torch.tanh(n28)
        n30 = n5
        n31 = torch.tanh(n30)
        n32 = torch.tanh(n29)
        n33 = (n31 + n32)
        n34 = torch.tanh(n33)
        v15 = self.conv8(n34)
        v16 = torch.sigmoid(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

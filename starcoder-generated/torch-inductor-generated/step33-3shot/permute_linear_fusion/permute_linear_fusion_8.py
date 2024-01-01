
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.conv3 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1.unsqueeze(1)
        v3 = v2.permute(0, 3, 1, 2)
        v4 = self.linear(v3)
        v5 = v4.to(torch.float32)
        w = v5.permute(0, 2, 3, 1)
        y1 = w * v1
        y2 = y1.unsqueeze(1)
        y3 = self.conv1(y2)
        y4 = y3.max(dim=-1)[0]
        y4 = torch.tanh(y4)
        v6 = torch.nn.functional.linear(y4, self.linear2.weight, self.linear2.bias)
        v7 = y4 * v6
        v8 = v7.sigmoid()
        y5 = v7.squeeze(3)
        y6 = y5.permute(0, 2, 1)
        y7 = self.conv2(y6)
        y8 = y7.max(dim=-1)[0]
        y8 = self.sigmoid(y8)
        y9 = -self.relu(93.305 - y8)
        y10 = -y9.unsqueeze(1)
        y11 = torch.nn.functional.linear(y10, self.linear3.weight, self.linear3.bias)
        y12 = 3 * y11
        y13 = self.relu(y12)
        y14 = self.conv3(y13)
        y15 = y14.flatten(1)
        y16 = self.sigmoid(y15)
        y17 = -66.767 + y16
        y18 = 3 * y17
        y19 = torch.tanh(y18)
        y20 = 29.523 * y19
        y21 = y20.unsqueeze(1)
        y22 = y21.squeeze(3)
        y23 = 10.57 + y22
        y24 = self.linear1(y23)
        y25 = self.sigmoid(y24)
        y26 = -53.611 + y25
        y27 = y26.unsqueeze(1)
        y28 = y27.squeeze(3)
        x3 = 2.477 * y28
        x2 = self.sigmoid(x3)
        x1 = x2
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 15, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(15, 14, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(15, 14, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(15, 12, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(12, 4, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(10, 5, 3, stride=1, padding=1)
        self.dense12 = torch.nn.Linear(36, 16)
        self.dense13 = torch.nn.Linear(18, 16)
        self.dense14 = torch.nn.Linear(58, 16)
    def forward(self, x7):
        v1 = self.conv0(x7)
        v2 = torch.nn.functional.tanh(v1)
        v3 = self.conv1(v2)
        v4 = torch.nn.functional.tanh(v3)
        v5 = self.conv2(v2)
        v6 = torch.nn.functional.tanh(v5)
        v7 = torch.nn.functional.relu(v4 + v6)
        v8 = v2 - v7
        v9 = self.conv3(v8)
        v10 = torch.nn.functional.softmax(v9)
        v11 = v7.flatten(start_dim=1)
        v12 = self.dense12(v11)
        v13 = torch.nn.functional.sigmoid(v12)
        v14 = self.dense13(x6)
        v15 = torch.nn.functional.tanh(v14)
        v16 = self.dense14(torch.cat((x1, v15), dim=-1))
        v17 = torch.nn.functional.relu(v13 + v16)
        v18 = v17.unsqueeze(2)
        v19 = torch.nn.functional.interpolate(v18, scale_factor=0.049132, mode='nearest')
        v20 = v19.squeeze(2)
        v21 = v20
        v22 = self.conv4(v20)
        v23 = v22 - -6.7033
        return v23
# Inputs to the model
x7 = torch.randn(1, 1, 64, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
        self.reshape = torch.nn.Flatten()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=[2], stride=[2])
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=[2], stride=[2])
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear2 = torch.nn.Linear(8, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        x2 = v2 + v1
        x3 = x2.reshape(torch.Size([1, 8]))
        x4 = x3.permute(0, 3, 2, 1)
        v4 = self.conv1(x4)
        v5 = self.conv2(v4)
        v6 = v4 + v5
        x6 = v6.permute(0, 3, 2, 1)
        x7 = x6.reshape(torch.Size([1, 8]))
        x8 = torch.sigmoid(x7)
        v8 = torch.softmax(x7, dim=-1)
        v9 = torch.stack((v8, x7), dim=-1)
        v9 = torch.max(v9, dim=-1)
        v9 = v9.values
        v10 = self.reshape(x8)
        v11 = self.flatten(v10)
        v11 = torch.tanh(v11)
        v12 = torch.sigmoid(self.linear2(self.dropout(v11)))
        v4 = v4 * x2
        x6 = x6 * x2
        v5 = torch.sigmoid(x6)
        x5 = torch.sum(v1 * v2, dim=-1, keepdim=True)
        x6 = x6 + x5
        x7 = x6.permute(0, 2, 1)
        v6 = x7.permute(0, 2, 1)
        v5 = torch.sigmoid(v6 + x7)
        v5 = v5.permute(0, 2, 1)
        v5 = torch.relu(v5)
        v5 = v5 + v6
        v7 = torch.relu(v5)
        v14 = torch.sigmoid(v5)
        v15 = torch.softmax(v5, dim=-1)
        v1 = x1.data.max(-1)[0]
        v2 = x1.data.min(-1)[0]
        v3 = x1.data.max(-1)[0]
        v4 = x1.data.min(-1)[0]
        v5 = torch.stack((v1, v2, v3, v4), dim=-1)
        v5 = torch.stack((v3, v1, v4, v2), dim=-1)
        v5 = v5 * x1.data
        v6 = v5.view(torch.Size([1, 8]))
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)

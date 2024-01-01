
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.sqrt(v2)
        x3 = x2 * v2
        v3 = torch.nn.functional.adaptive_avg_pool1d(x2, 1)
        v4 = torch.nn.functional.adaptive_avg_pool1d(x3, 1)
        v5 = torch.nn.functional.sigmoid(v3)
        v6 = torch.nn.functional.sigmoid(v4)
        v7 = (v5 >= v6).permute(0, 2, 1)
        x4 = torch.nn.functional.relu(x2)
        v8 = x4 * v7
        x5 = torch.nn.functional.max_pool1d(v8, 2)
        v9 = x5.permute(0, 2, 1)
        v10 = torch.nn.functional.max_pool1d(v2, 2)
        x6 = torch.nn.functional.conv1d(v9, v10)
        x7 = torch.nn.functional.softmax(x6, -1)
        x8 = x7 * v8
        x9 = torch.nn.functional.softmax(v8, -1)
        v11 = torch.cat((x8, x9), 1)
        x10 = torch.nn.functional.log_softmax(v11, -1)
        x11 = x10 * v9
        x12 = torch.nn.functional.sigmoid(v8)
        x13 = torch.cos(x12)
        x14 = x12 + v8
        x15 = torch.sinh(v8)
        x16 = torch.pow(x14, 2)
        x17 = torch.pow(x15, 2)
        x18 = torch.pow(x12, 2)
        x19 = torch.pow(x13, 2)
        v12 = torch.stack((x17, x18, x19))
        x20 = torch.sqrt(v12)
        v13 = torch.stack((x8, x9))
        x21 = torch.argmax(v13, -1)
        v14 = torch.stack((x8, x9))
        x22 = torch.argmin(v14, -1)
        return torch.stack((x21, x22))
# Inputs to the model
x1 = torch.randn(1, 2, 2)

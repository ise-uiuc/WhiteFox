
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
    def forward(self, x1):
        v1 = torch.abs(self.conv1(x1))
        v2 = torch.sigmoid(v1)
        v3 = torch.mean(v2, dim=0)
        v4 = self.conv1(x1)
        v5 = torch.abs(v4)
        v6 = torch.sigmoid(v5)
        v7 = torch.mean(v6, dim=0)
        v8 = torch.cat((v3, v7), 0)
        v9 = self.fc1(v8)
        v10 = torch.sigmoid(v9)
        v11 = torch.nn.functional.dropout(v10, p=0.5, training=True)
        v12 = self.fc2(v11)
        return v12
# Inputs to the model
a = torch.randn(1, 1, 28, 28)

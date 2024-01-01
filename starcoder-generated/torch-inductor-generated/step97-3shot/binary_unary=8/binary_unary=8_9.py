
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.view(-1, 1, 4, 16, )
        v2 = v1.view(-1, 256)
        v3 = v2.view(4, 4)
        v4 = torch.relu(v3)
        v5 = v2.view(4, 4)
        v6 = self.conv1(x1)
        v7 = v5 + v6
        v8 = v7.view(-1, 4, 4, 16, ).view(-1, 256)
        v9 = torch.relu(v8)
        v10 = self.conv1(x1)
        v11 = v10 + v9
        v12 = v11.view(-1, 4, 4, 16, ).view(-1, 256)
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)

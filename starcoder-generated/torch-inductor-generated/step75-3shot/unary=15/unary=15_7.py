
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.fc1 = torch.nn.Linear(400, 512)
        self.fc2 = torch.nn.Linear(512, 128)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.reshape(v2, (1, -1))
        v4 = self.fc1(v3)
        v5 = torch.relu(v4)
        v6 = self.fc2(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 20, 128, 128)

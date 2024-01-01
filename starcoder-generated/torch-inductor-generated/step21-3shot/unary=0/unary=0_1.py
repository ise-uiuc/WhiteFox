
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 10, stride=1)
        self.conv2 = nn.Conv2d(32, 10, 5, stride=1)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10*11*11, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = x.view(-1, 10*11*11)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)
# Inputs to the model
x = torch.randn(1, 1, 88, 187)

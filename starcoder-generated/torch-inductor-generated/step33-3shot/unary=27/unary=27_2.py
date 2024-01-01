
class MnistModel(torch.nn.Module):
    def __init__(self, min, max):
        super(MnistModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.max = min
        self.min = max

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.clamp_min(x, self.min)
        x = torch.clamp_max(x, self.max)
        return F.log_softmax(x, dim=1)
min = 0.0
max = 1.0
# Inputs to the model
x = torch.randn(1, 1, 28, 28, requires_grad=True)

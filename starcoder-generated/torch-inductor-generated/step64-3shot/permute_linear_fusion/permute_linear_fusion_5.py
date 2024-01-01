
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = Flatten()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.tanh1 = torch.nn.Tanh()
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.dropout1 = torch.nn.Dropout2d(p=0.2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout2d(p=0.5)
        self.flatten1 = Flatten()
        self.linear1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(500, 10)
    def forward(self, x):
        x = self.tanh1(self.conv1(self.flatten(x)))
        x = self.max_pool(self.dropout1(x))
        x = self.tanh1(self.conv2(x))
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        return x
# Inputs to the model
x = torch.randn(1, 1, 28, 28)



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(32, 32)
        self.relu_1 = nn.ReLU()
        self.t5 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.t5(x)
        x = self.softmax(x)
        return x


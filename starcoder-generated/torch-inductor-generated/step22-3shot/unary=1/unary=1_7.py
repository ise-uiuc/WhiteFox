
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2048, 1024)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear3 = torch.nn.Linear(512, 256)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear4 = torch.nn.Linear(256, 2)

    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = self.relu1(x2)
        x4 = self.linear2(x3)
        x5 = self.relu2(x4)
        x6 = self.linear3(x5)
        x7 = self.relu3(x6)
        x8 = self.linear4(x7)
        return x8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048)

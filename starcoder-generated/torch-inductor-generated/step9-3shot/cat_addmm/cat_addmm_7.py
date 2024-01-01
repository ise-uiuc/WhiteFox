
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 8)
        self.act2 = torch.nn.ReLU()
        self.fcout = torch.nn.Linear(40, 10)
 
    def forward(self, x1):
        x = x1
        x = self.fc1(x)
        x = self.act1(x)
        y = x
        x = self.fc2(x)
        x = self.act2(x)
        z = y + x
        c = torch.cat([z], dim=0)
        last = self.fcout(c)
        return last

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)

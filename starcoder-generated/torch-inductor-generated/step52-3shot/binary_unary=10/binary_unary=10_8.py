
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(inplace=True))
 
    def forward(self, x1, x2):
        y = torch.cat((x1, x2), dim=1)
        y0 = self.fc1(y)
        y0 = y + y0
        y1 = self.fc2(y0)
        y1 = y0 + y1
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 8)

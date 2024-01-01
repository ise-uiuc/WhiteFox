
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(300, 300)
        self.fc2 = torch.nn.Linear(300, 300)
        self.fc3 = torch.nn.Linear(300, 300)
        self.fc4 = torch.nn.Linear(300, 100)
    def forward(self, input):
        x = self.fc1(input)
        y = self.fc2(x)
        z = self.fc3(input)
        a = x.add(y)
        b = a.add(self.fc3(input))
        c = z.add(self.fc4(input))
        return b.tanh()
# Inputs to the model
input = torch.randn(1, 300)

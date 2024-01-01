
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l1 = torch.nn.Linear(1, n)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y1 = self.l1(x)
        y2 = self.relu(y1)
        y3 = self.l1(y2)
        y4 = torch.cat((y1, y2))
        y5 = torch.tanh(y4)
        return y5

# Initialize the model
n = 2
m = Model(n)

# Inputs to the model
x = torch.randn(1, 1)

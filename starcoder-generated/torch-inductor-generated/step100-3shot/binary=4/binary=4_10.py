
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
 
    def forward(self, x, y, z):
        x = self.fc1(x)
        y1 = x + y
        z1 = x + z
        return y1, z1
 
class Model2(torch.nn.Module):
    def __init__(self, m, n):
        super(Model2, self).__init__()
        self.model = m
        self.fc1 = torch.nn.Linear(n, 10)
 
    def forward(self, x):
        y = torch.randn(3, 2)
        z = torch.randn(3, 4)
        return self.model(x, y, z)

# Initializing the model
m = Model1()
m1 = Model2(m, 10)

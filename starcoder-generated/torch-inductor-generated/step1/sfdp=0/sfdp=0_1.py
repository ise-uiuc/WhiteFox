
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(8, 16)
        self.key = torch.nn.Linear(8, 16)
        self.value = torch.nn.Linear(8, 16)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x):
        a1 = self.query(x)
        a2 = self.key(x)
        a3 = a2.transpose(-2, -1)
        a4 = torch.matmul(a1, a3)
        a5 = a4.div(math.sqrt(16.0))
        a6 = self.softmax(a5)
        a7 = self.value(x)
        v = torch.matmul(a6, a7)
        return v

# Initializing the model to randomly choose whether adding dropout between attention layers or not)
m = Model()

# Inputs to the model
x = torch.randn(1, 8)

if 0!= randint(0, 1):
    m.query = torch.nn.Sequential(m.query, torch.nn.Dropout(0.15922771904468536))
    m.key = torch.nn.Sequential(m.key, torch.nn.Dropout(0.07502236720790863))
    m.value = torch.nn.Sequential(m.value, torch.nn.Dropout(0.9361229133605957))

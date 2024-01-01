
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x2):
        v0 = self.linear(x2)
        v1 = self.dropout(v0)
        return v1

# Initializing the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = lowmem_dropout(v1, p=0.5, training=True, inplace=False)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 2)

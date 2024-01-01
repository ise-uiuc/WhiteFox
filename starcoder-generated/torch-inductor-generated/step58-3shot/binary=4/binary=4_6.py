
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(16, 16)
        self.layer2 = torch.nn.Linear(16, 8)
        self.tanh = torch.nn.Tanh()
     
    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.tanh(l1)
        l3 = self.layer2(l2)
        a1 = l3 + l1
        return a1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(8, 16)


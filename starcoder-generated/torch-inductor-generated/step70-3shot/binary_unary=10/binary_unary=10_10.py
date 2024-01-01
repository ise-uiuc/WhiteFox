
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(15, 80)
        self.linear2 = torch.nn.Linear(80, 20)
 
    def forward(self, x1):
        y = self.linear1(x1)
        y = self.linear2(y)
        y = y + torch.tanh(y)
        y = F.relu(y)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2)

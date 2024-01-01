
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 128)
        self.linear2 = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        t1 = torch.nn.functional.relu(self.linear1(x1)), self.linear2(x1))
        t2 = torch.nn.functional.relu(self.linear2(x1))
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 128)

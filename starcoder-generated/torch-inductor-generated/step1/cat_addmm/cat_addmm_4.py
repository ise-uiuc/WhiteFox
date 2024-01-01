
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(4, 2)
 
 
    def forward(self, x):
        v1 = self.linear1(x)
        v4 = self.linear2(v1)
        v3 = torch.cat((v1.unsqueeze(0), v4.unsqueeze(0)), 0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)

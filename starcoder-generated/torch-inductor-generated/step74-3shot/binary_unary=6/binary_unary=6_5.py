
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(17, 10)
        self.linear_2 = torch.nn.Linear(10, 20)
 
    def forward(self, x):
        v5 = self.linear_2(self.linear_1(x))
        v4 = v5.sum(dim=0)
        v3 = v5 - v4
        return v3 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(13, 17)

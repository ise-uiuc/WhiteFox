
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1, x2=1, x3=2, x4=3, x5=4):
        x = torch.cat([x1, x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1), x5.unsqueeze(1)], dim=1)  
        v2 = self.linear(x)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.zeros(2, 5)
x2 = torch.tensor([1, 2])
x3 = torch.tensor([1, 2])
x4 = torch.tensor([1, 2])
x5 = torch.tensor([1, 2])

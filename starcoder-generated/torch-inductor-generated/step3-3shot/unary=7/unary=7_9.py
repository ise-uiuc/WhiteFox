
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32,10)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 * torch.min(torch.max(v1+3,torch.tensor([0])),torch.tensor([6]))
        v3 = v2/6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(32,32)

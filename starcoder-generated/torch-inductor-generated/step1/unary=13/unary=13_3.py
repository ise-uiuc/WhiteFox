
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 16)
        self.linear2 = torch.nn.Linear(16, 8)
 
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)

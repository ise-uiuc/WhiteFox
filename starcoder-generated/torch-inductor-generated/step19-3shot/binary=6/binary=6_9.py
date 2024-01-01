
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(512, 20)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        res = v1 - 100
        return v1
 
# Initializing the model
m = Model()
 
# Input to the model
x1 = torch.randn(1, 512)

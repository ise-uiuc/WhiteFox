
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 200)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
    
m = Model()

# Inputs to the model
x = torch.randn(1, 100)

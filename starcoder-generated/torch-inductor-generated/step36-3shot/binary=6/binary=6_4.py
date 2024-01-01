
class Model(torch.nn.Module):
    def __init__(self, size1,size2,size3,size4):
        super().__init__()
        self.fc = torch.nn.Linear(size1, size2)
        self.size3 = size3
        self.size4 = size4
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - self.size3
        v3 = v1 - self.size4
        return v2, v3

# Initializing the model
m = Model(10,10,10,10)

# Inputs to the model
x1 = torch.randn(128, 10)

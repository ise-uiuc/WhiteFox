
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = torch.sigmoid(m1)
        return m2

# Initializing the model
m=Model()

#Inputs to the model
x1 = torch.randn(1, 8)

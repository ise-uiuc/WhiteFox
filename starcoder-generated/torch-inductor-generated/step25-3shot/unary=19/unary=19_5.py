
class Model(torch.nn.Module): # a neural network containing a single linear layer and a Sigmoid layer
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = F.sigmoid(v1)
        return v2

# Initializing and validating the model
m = Model()
print(m(torch.tensor([[1.84601928, 1.91643565]])))

# Inputs to the model
x1 = torch.randn(4, 2)

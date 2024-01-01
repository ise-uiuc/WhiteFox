
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)
 
    def forward(self, x1):
        x2 = x1.view(-1, 4)
        v1 = self.fc(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 4) # This model works only with a tensor where the size of dimension 1 is at least 6.


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 16)
 
    def forward(self, x1):
        return F.relu(self.fc(x1) + torch.rand(1,16))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)

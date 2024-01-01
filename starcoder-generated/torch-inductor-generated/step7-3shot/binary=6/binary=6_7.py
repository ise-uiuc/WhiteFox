
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(192, 10, bias=False)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - torch.ones(1, 10).to(device)
        return v2

# Initializing the model
m = Model()

# Inputs to the model. Please select the value of other that makes v2 have zero norm within reasonable tolerance.
x1 = torch.randn(1, 192)
other = 0.

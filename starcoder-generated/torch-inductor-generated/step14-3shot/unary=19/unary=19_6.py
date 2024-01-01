
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(200, 400)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.nn.functional.sigmoid(v1)
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 200)

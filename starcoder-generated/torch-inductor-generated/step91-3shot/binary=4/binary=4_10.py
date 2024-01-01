
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 10)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 + other
        return v2

# Input to the model
x = torch.randn(1, 20)
other = torch.randn(1, 10)
# Initializing the model
m = Model()

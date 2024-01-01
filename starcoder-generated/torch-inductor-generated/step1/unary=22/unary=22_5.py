
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()
x = torch.randn(1, 8)

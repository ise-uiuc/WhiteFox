
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 1, bias=True)
        self.fc2 = torch.nn.Linear(1, 8, bias=False)
        self.other = torch.randn(1, 1)
 
    def forward(self, x):
        v1 = self.fc1(x).flatten()
        v2 = v1 + self.other
        v3 = self.fc2(v2)
        return v3

# Initializing the model
m = Model()
x = (torch.randn(1, 4))

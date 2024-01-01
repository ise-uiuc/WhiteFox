
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 4)
 
    def forward(self, x1, x2, other=None):
        v1 = self.fc(x1)
        v2 = v1 + x2
        if other is not None:
            v2 += other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 20)
x2 = torch.randn(10, 4)
other = torch.tensor([[1, 1, 1, 1]]).expand_as(x2)

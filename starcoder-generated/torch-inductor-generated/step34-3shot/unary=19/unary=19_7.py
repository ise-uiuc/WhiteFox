
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3*4*4*4, 128)
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.concat((x1, x2, x3, x4))
        v2 = v1.view(-1, 3*4*4*4)
        v3 = self.linear(v2)
        v4 = torch.sigmoid(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3, 4, 4)
x2 = torch.randn(4, 3, 4, 4)
x3 = torch.randn(4, 3, 4, 4)
x4 = torch.randn(4, 3, 4, 4)


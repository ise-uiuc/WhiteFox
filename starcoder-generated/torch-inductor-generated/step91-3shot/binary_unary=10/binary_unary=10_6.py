
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 + x1
        v3 = nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(512)
x2 = torch.randn(512, 1024)

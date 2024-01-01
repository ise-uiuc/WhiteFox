
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(800, 400)
 
    def forward(self, x2):
        v7 = F.relu(self.linear(x2))
        v8 = v7 + 3
        v9 = F.relu(0, 6)
        v10 = v9 / 6

# Initializing the model
m = Model()
 
# Inputs to the model
x2 = torch.randn(1, 800)

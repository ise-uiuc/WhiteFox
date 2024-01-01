
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 40)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 31459
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.rand(25, 100)

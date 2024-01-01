
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = torch.nn.Linear(3, 1)
 
    def forward(self, t1):
        v1 = self.liner(t1)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 3, 64, 64)

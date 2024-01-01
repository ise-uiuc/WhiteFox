
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(20, 20),
            nn.Sigmoid(),
        )
 
    def forward(self, x0):
        v0 = x0.squeeze()
        v1 = self.layers(v0)
        v2 = v0 * v1
        return v2.unsqueeze(0)

# Initializing the model
m = Model()

# Inputs to the model
input0 = torch.randn(1, 20)

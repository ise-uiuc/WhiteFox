
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(180, 64)
 
    def forward(self, input):
        v1 = self.linear1(input)
        v2 = torch.clamp_min(v1, 0.5)
        v3 = torch.clamp_max(v2, 0.7071067811865476)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 180)

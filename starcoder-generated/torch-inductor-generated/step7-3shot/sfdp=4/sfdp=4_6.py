
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v0, v1):
        v2 = (v0 @ v1.transpose(-2, -1)) / math.sqrt(v0.size(-1))
        v3 = v2 + v1.unsqueeze(2)
        v4 = torch.softmax(v3, dim=-1)
        v5 = v4 @ v2
        return v5

# Initializing the model
m = Model()

# Inputs to the model
v0 = torch.randn(3, 4, 2)
v1 = torch.randn(3, 5, 4)

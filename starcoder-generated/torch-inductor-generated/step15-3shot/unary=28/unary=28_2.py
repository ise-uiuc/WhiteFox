
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = v1 * 0.5
        v3 = torch.sigmoid(v2)
        v4 = v3 * 0.7071067811865476
        v5 = torch.sigmoid(v4)
        v6 = v5 * 0.7071067811865476
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

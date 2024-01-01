
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, min_value, max_value):
        v1 = torch.clamp(x1, min=min_value, max=max_value)
        v2 = v1 * 0.7071067811865476 - 0.25
        v3 = v2 * v2 - 0.5
        v4 = v1 * v3 + 0.5
        v5 = v4 + 2
        return v5

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.ones(1, 8, 64, 64)
min_value = 1.8
max_value = 2.5

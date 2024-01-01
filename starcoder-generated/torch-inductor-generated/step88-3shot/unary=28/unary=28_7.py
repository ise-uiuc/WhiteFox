
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.from_numpy(weights).float())
        v2 = torch.clamp_min(v1, -1.25)
        v3 = torch.clamp_max(v2, 1.25)
        return v3

weights = np.random.random((3, 5)).astype(np.float32)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5)


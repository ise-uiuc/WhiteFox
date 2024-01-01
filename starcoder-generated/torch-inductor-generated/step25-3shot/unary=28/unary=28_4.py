
class Model(torch.nn.Module):
    def __init__(self, min_value=-5, max_value=10):
        super().__init__()
        self.weight = torch.rand(1, 8, dtype=np.float32)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = torch.tensordot(x1, self.weight, dims=1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 3)

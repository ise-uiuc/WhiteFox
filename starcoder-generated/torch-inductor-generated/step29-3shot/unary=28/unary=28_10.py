
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.rand([1, 1024], dtype=torch.float32), torch.rand([1024], dtype=torch.float32))
        v2 = torch.nn.functional.clamp(v1, min=self.min_value, max=self.max_value)
        v3 = torch.nn.functional.relu(v2)
        return v3
 
# Initializing the model
m = Model(-0.5, 1)

# Inputs to the model
x1 = torch.randn(1, 1024, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, 0, 0.5)
        return v2

# Initializing the model
__seed_value = 456677__
torch.manual_seed(__seed_value)
# We'll try 5 randomly different configurations:
for i in range(5):
    m = Model()
    # Inputs to the model
    x1 = torch.randn(1, 16)
    
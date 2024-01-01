
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([x] * 32, 1)
# Inputs to the model
x = -torch.rand(12, 16) # Only supports float16 and float32

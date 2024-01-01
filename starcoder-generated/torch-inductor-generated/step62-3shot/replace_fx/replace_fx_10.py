
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = random.random()
        return v1
# Inputs to the model
x1 = torch.randn(32, 32)

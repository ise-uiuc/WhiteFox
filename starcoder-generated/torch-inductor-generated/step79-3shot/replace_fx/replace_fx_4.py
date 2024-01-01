
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randint_like(x1, low = -128, high = 128)
        x3 = torch.randint_like(x1, low = -128, high = 128)
        x4 = torch.randint_like(x2)
        x5 = torch.randint_like(x2)
        return x1
# Inputs to the model
x1 = torch.randint(low = -100, high = 100, size=(1, 2, 2, 3), dtype = torch.int32)

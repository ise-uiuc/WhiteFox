
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v2 = torch.rand_like(x1)
        v4 = torch.nn.functional.dropout(x1, 0.3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(512, 512)

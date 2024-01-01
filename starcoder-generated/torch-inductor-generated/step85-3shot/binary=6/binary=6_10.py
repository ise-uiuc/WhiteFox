
class LinearAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.nn.functional.linear(x, torch.randn(6, 4), bias=torch.randn(6))
        v2 = v1 - torch.tensor([1, 2, 3, 4, 5, 6])
        return v2

# Initializing the model
m = LinearAddModel()

# Inputs to the model
x = torch.randn(6, 4)

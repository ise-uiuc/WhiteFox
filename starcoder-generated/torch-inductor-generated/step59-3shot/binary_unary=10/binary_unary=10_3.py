
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x2):
        v1 = torch.nn.functional.linear(
            x2,torch.tensor([0.5, 1.0], dtype=torch.float32)
        )
        v2 = torch.nn.functional.linear(
            x2,torch.tensor([0.5, 1.0], dtype=torch.float32)
        )
        v3 = v1 + 2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = ModelNew()

# Inputs to the model
x2 = torch.randn(1, 3, 8, 8)

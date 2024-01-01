
class Model(torch.nn.Module):
    def forward(self, x1, other=None):
        if other is None:
            other = torch.tensor([0, 0, 0], dtype=x1.dtype, device=x1.device)
        return torch.relu(x1 + other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):

    def forward(self, x1, x2):
        v1 = x1 @ x2
        v2 = v1 - x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
__all__ = ["Model"]

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(5, 1)

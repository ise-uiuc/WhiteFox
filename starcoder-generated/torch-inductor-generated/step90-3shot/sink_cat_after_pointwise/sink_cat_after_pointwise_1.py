
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z, k):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        z = torch.flatten(z, start_dim=1, end_dim=-1)
        return torch.relu(torch.cat((x, y, z, k), dim=1))
# Inputs to the model
x = torch.randn(1, 2, 3, 5)
y = torch.randn(1, 2, 3, 4)
z = torch.randn(1, 2, 3, 3)
k = torch.randn(1, 2, 3, 2)

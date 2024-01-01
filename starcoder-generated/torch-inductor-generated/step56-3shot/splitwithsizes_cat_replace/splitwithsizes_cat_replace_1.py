
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Identity()
    def forward(self, v1, p1, p2):
        return p1 + p2 + torch.relu(v1)
# Inputs to the model
x1 = torch.randn(1, 32, 4, 4)
p1 = torch.randn(1, 5, 128, 128)
p2 = torch.randn(1, 2, 128, 128)

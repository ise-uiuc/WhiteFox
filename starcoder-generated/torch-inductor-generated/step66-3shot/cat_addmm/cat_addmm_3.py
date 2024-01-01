
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 6)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        print(torch.view_as_real(x))
        return x
# Inputs to the model
x = torch.rand(2, 2) + 1j * torch.rand(2, 2)

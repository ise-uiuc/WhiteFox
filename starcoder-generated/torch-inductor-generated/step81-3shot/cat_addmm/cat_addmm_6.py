
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, dim=2)
        x = torch.flatten(x, start_dim=2)
        x = torch.stack((x, x, x, x, x, x), dim=2)
        x = torch.swapaxes(x, dim1=0, dim2=2)
        x = torch.flatten(x, start_dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)

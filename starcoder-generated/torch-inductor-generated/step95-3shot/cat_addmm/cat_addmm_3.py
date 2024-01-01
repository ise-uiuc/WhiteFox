
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        l1 = self.layers(x)
        x = x.unsqueeze(dim=1)
        l2 = self.layers(x)
        x = x.unsqueeze(dim=1)
        l3 = self.layers(x)
        x = x.flatten(start_dim=1)
        l4 = self.layers(x)
        x = x.flatten(end_dim=1)
        return torch.cat((l1, l2, l3, l4), dim=1)
# Inputs to the model
x = torch.randn(2, 2)

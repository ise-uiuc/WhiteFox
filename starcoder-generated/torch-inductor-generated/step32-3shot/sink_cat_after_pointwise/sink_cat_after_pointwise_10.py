
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x], dim=0)
        x = torch.sigmoid(x) if torch.numel(x) == 1 else torch.sigmoid(x)
        return torch.relu(x)
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.tanh(x)
        x = x.unsqueeze(dim=1)
        x = torch.cat([x, x, x], dim=-1)
        return torch.relu(x.flatten(begin_dim=1))
# Inputs to the model
x = torch.randn(2, 3, 4)

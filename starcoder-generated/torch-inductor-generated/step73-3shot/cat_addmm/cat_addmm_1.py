
class Model(nn.Module):
    def __init__(self, n_features, out_dim):
        super().__init__()
        self.linear = nn.Linear(n_features, out_dim)
    def forward(self, x):
        x = self.linear(x)
        x = torch.stack(4 * [x], dim=1)
        x = x.flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 8)

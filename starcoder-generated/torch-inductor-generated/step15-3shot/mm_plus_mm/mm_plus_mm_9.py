
class Model(torch.nn.Module):
    def forward(self, X):
        return torch.mm(X, 2)
# Inputs to the model
X = torch.randn(100, 100)

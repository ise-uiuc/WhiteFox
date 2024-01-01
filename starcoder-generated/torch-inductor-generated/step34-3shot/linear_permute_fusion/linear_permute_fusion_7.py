
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, X):
        yhat_p1 = self.linear(X).add(1).pow(-2).prod(-1).clamp(min=1e-15).repeat(2, 2).flatten().add(1.).reciprocal().add(1e-15)
        yhat = torch.stack((yhat_p1, yhat_p1), dim=1)
        return yhat
# Inputs to the model
X = torch.randn(1, 2, 2)

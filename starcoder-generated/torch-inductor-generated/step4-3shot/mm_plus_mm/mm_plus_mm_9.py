
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, W, R, S):
        v1 = torch.mm(X, W)
        v2 = X * R + v1
        v3 = S + v2
        return v3
# Inputs to the model
X = torch.randn(4, 35)
W = torch.randn(35, 10)
R = torch.randn(10, 10)
S = torch.randn(35, 10)

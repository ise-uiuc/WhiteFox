
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        c1 = torch.nn.functional.dropout(X, p=0.2)
        return 1
# Inputs to the model
X = torch.rand([1, 1])

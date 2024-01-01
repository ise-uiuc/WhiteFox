
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, W, H):
        N, C, H, W = X.shape
        padding_tensor = [[0, 0], [0, 0], [padding_up, padding_down], [padding_left, padding_right]]
        X_padded = F.pad(X, padding_tensor, "constant", 0)
        # X_padded = F.pad(X, (0, 0, 0, 0, padding_up, padding_down, padding_left, padding_right))
        return X_padded
# Inputs to the model
X = torch.randn(4, 4, 5, 5)
W = 2
H = 2

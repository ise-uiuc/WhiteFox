
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 7, 3, stride=(2, 2))
    def forward(self, X):
        X = self.conv_transpose(X)
        X = X + 2
        return X
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)

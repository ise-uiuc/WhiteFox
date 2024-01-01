
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 5, 2, stride=2)
    def forward(self, features):
        out_features = self.conv_transpose(features)
        return out_features
# Inputs to the model
x = torch.randn(1, 5, 5, 5)

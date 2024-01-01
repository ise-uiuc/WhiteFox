
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.ConvTranspose2d(64, 64, (2,2), stride=(2, 2), padding=(0, 0))(x1)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = v2 * v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), padding=(2, 2))
    def forward(self, x1):
        v1 = self.transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)

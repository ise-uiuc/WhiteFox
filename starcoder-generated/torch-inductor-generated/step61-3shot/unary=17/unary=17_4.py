
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.ConvTranspose2d(4, 32, (1, 1), stride=(1, 1), groups=4)(x1)
        v2 = torch.relu(v1)
        v3 = torch.nn.ConvTranspose2d(32, 16, 1, stride=1, padding=0)(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.nn.ConvTranspose2d(16, 32, (3, 3), stride=(1, 1), output_padding=(1, 1), groups=4)(v4)
        v6 = torch.sigmoid(v5)
        v7 = torch.nn.ConvTranspose2d(32, 64, kernel_size=(1, 1), stride=(2, 2), padding=0, output_padding=(0, 0))(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 1024, 512)

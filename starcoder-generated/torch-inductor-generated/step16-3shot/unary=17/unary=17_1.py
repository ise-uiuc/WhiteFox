
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), padding=int(sys.argv[4]), stride=int(sys.argv[5]))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)

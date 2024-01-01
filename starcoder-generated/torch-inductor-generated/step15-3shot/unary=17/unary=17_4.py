
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2)
        self.max_pool = torch.nn.MaxPool2d(4, return_indices=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3,v4 = torch.max_pool(v2, 4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 512, 512)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 2, 6, kernel_size=(2, 3), stride=(1, 1))
        self.conv2 = torch.nn.ConvTranspose2d(2, 1, 3, padding=(2, 2), output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
        
# Inputs to the model
x1 = torch.randn(1, 1, 25, 18)

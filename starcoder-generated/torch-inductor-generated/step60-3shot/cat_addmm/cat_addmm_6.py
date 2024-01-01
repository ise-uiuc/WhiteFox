
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Conv2d(16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Conv2d(32, out_channels=64, kernel_size=(3, 3), padding=1)
        )
    def forward(self, x):
        x = self.layers(x)  # The second to last dimension of x is the channel dimension
        x = torch.flatten(x, start_dim=0, end_dim=1) # Delete all dimensions between 0 and 1
        x = F.relu(x)  # The activation function should not change the shape of x
        return x
# Inputs to the model
x = torch.randn(16, 1, 5, 5)

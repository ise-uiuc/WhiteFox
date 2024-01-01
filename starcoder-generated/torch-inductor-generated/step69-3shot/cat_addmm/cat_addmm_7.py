
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(8, 8), stride=(15, 15), padding = (3, 3))
        self.conv2 = nn.Conv2d(24, 64, kernel_size = (5, 5), stride=(2, 2), padding = (4, 4))
        self.conv3 = nn.Conv2d(10, 64, kernel_size = (3, 3), stride=(1, 1), padding = (2, 2))
        self.conv4 = nn.Conv2d(1, 32, kernel_size = (7, 7), stride=(1, 1), padding = (3, 3))
        self.linear1 = nn.Linear(128, 10)
    def forward(self, x):
        # x shape: 16 x 3 x 32 x 32
        x = self.conv1(x)
        # x shape: 16 x 96 x 13 x 13
        x = self.conv2(x)
        # x shape: 16 x 64 x 6 x 6
        x = self.conv3(x)
        # x shape: 16 x 64 x 4 x 4
        x = self.conv4(x)
        # x shape: 16 x 32 x 2 x 2
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # x shape: 16 x 128
        x = self.linear1(x)
        # x shape: 16 x 10
        return x
# Inputs to the model
x = torch.randn(16, 1, 32, 32)

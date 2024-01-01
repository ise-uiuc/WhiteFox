
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 1, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv_transpose1(x1)

        self.fc_1 = torch.nn.Linear(8, 4)

        self.sigmoid1 = torch.nn.Sigmoid()

        v2 = self.fc_1(v1)
        v3 = self.sigmoid1(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 3, 49, 49)

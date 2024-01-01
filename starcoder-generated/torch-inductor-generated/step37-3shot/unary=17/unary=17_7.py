
class Model(nn.Module):
    def __init__(self, num_classes=340):
        super(Model, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1) # The "zeroed" padding was used to generate different input tensors
        self.conv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(4, 1, 3, stride=1)
        self.dense = nn.Linear(1, num_classes)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4) # The "sigmoid" activation function was applied to the output of the transposed convolution
        v6 = v5.view(v5.shape[0], -1)
        v7 = self.dense(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

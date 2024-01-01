
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(16, 20, 5, stride=(2, 2), padding=(2, 2), output_padding=(2, 0))
        self.conv3 = torch.nn.ConvTranspose2d(20, 20/2, 5, stride=((1,2),(3,5)), padding=(4,1), output_padding=(0,0))
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = torch.max_pool2d(v1, 4, 2, 1)
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 35, 35)

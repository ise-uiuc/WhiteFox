
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, padding=0)
    def forward(self, x1):
        x = self.conv_transpose(x1)
        x_relu = F.relu(x)
        x_relu_mean = torch.mean(x_relu)
        x_relu_mean_sigmoid = torch.sigmoid(x_relu_mean)
        return x_relu_mean_sigmoid
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

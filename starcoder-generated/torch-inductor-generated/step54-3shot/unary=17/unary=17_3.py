
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 9, (5, 5), stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 23, 23)

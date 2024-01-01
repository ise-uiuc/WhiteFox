
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(16, 50, 1, stride=1, padding=0)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(51, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        q1 = self.conv_transpose_4(x1)
        q2 = torch.sigmoid(q1)
        q3 = q1 * q2
        q4 = self.conv_transpose_5(q3)
        q5 = torch.sigmoid(q4)
        q6 = q3 * q5
        return q6
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)


class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(17, 6, 4, stride=2, padding=1)
        self.linear = torch.nn.Linear(1432, 1024)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.linear(self.conv_transpose(x1).view(1, -1)))
        v2 = v1 * 0.3333448449516386
        v3 = v1 * 0.5563404776059723
        v4 = torch.nn.functional.relu(0.744832408111572 * v3)
        v5 = v4 + 1.192092871791931e-06
        v6 = v2 * v4
        return v6
model = Model()
# Inputs to the model
x1 = torch.randn(1, 17, 23, 11, 29)

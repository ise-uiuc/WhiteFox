
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(2, 8, 3, stride=2, padding=1)
    def forward(self, input_1):
        identity = input_1
        features = self.conv2d_0(input_1)
        return identity, features
# Inputs to the model
input_1 = torch.randn(1, 2, 9, 12)

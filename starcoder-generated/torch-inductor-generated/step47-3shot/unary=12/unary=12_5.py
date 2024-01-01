
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, (3, 3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(4, 1, (1, 1), stride=(1, 1))
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        ret = x.reshape(x.shape[0], -1)
        return ret
# Inputs to the model
input = torch.rand(1, 4, 32, 32)

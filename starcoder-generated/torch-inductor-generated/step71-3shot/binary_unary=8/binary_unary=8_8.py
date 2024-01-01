
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=0)
    def forward(self, input, param1, flag):
        v1 = self.conv(input)
        output = F.relu(v1)
        if flag == True:
            output = output + param1
        return output
# Inputs to the model
input = torch.randn(1, 3, 32, 32)
param1 = torch.randn(1, 8, 32, 32)
flag = True

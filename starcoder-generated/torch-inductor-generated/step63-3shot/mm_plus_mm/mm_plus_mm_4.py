
class Model(torch.nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 10, 5, 1)

    def forward(self, input1, input2, input3, ):
        v0 = self.conv(input1)
        v1 = v0.shape
        v2 = torch.randn(v1 + v1)
        v3 = v2.reshape(v1)
        v4 = v2.reshape(v1)
        v5 = self.conv(input1)
        v6 = v4 + self.conv(v3)
        v7 = v5 - v6
        v8 = input2 + v5
        v9 = input3 + v7
        return v8 + v9
# Inputs to the model
input1 = torch.randn(5, 3, 12, 12)
input2 = torch.randn(10)
input3 = torch.randn(10)


class Model(torch.nn.Module):
    def forward(self, data):
        data1 = data.reshape(2, 2)
        data2 = data.reshape(1, 2)
        result = (data1 + data2).reshape(2, -1)
        return result
# Inputs to the model
data = torch.randn(200, 2)

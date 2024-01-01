
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 15, 10, 10)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.cat([torch.relu(torch.tanh(torch.sigmoid(torch.abs(torch.pow(x, 2.0))))), x])
    def forward(self, x):
        return self.op
# Inputs to the model
x = torch.randn(2, 3, 4)

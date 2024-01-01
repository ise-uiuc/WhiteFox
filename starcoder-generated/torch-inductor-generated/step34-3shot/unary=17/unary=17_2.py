
class Model(torch.nn.Module):
    def __init__(self):
        super(Model_v1, self).__init__()
        self.conv = torch.nn.ConvTranspose1d(2, 3, kernel_size=(2,), stride=(2,))
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.relu(x1)
        x3 = self.linear(x1)
        return x3
# Inputs to the model
x = torch.Tensor(1, 2, 13, 13)

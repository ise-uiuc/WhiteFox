
class Model_v3(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 256)
        self.bilinear1 = torch.nn.Bilinear(128, 256, hidden_size)
        self.linear1.weight.data.fill_(0.001)
        self.bilinear1.weight.data.fill_(0.001)
    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.bilinear1(x, out1)
        out = torch.bmm(torch.transpose(x, 0, 1), out2)
        return out
# Inputs to the model
x = torch.randn(48, 128) # input tensor
hidden_size = 32          # hidden size

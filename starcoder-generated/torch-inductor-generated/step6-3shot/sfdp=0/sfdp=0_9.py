
class Model(torch.nn.Module):
    def __init__(self, d_key, d_value, d_model=512):
        super(Model, self).__init__()
        self.scale = 1 / (d_key ** 0.5)
        self.softmax = torch.nn.Softmax(-1)
        self.linear = torch.nn.Linear(d_key, d_value)
 
    def forward(self, x1, x2, x3):
        x2 = self.linear(x2)
        x2 = self.softmax(x2 * self.scale)
        x3 = self.linear(x3)
        x3 = x3.transpose(3, 2)
        return torch.matmul(x1, x2) + torch.matmul(x1, x3.transpose(-1, -2))


m = Model(512, 512)

# Inputs to the model
x1 = torch.randn(2, 16, 512)
x2 = torch.randn(3, 512).transpose(1, 0)
x3 = torch.randn(3, 512).transpose(1, 0)

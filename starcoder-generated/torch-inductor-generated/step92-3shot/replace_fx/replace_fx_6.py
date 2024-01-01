
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(16, 32, kernel_size=(3, 3))
    def forward(self, x, y):
        x = self.encoder(x)
        x = F.dropout(x, 0.2)
        y = torch.rand_like(x)
        y = torch.nn.Softmax()(y)
        z = x - y
        return z
# Inputs to the model
y = torch.randn(1, 16, 32, 32)
x = torch.randn(1, 16, 32, 32)

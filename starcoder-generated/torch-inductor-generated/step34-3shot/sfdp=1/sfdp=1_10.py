
class Model(torch.nn.Module):
    def __init__(self, num_heads=1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x1, x2, x3):
        y1 = x1.matmul(x2.transpose(-2, -1))
        y2 = y1 / self.num_heads
        y3 = y2.softmax(dim=-1)
        y4 = torch.nn.functional.dropout(y3)
        out = y4.matmul(x3)
        return out

# Initializing the model
m = Model(num_heads=12)

# Inputs to the model
x1 = torch.randn(1, 12, 1024)
x2 = torch.randn(1, 12, 2048)
x3 = torch.randn(1, 2048)

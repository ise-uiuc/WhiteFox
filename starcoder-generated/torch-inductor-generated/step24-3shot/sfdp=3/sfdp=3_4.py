
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, dropout_p=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.scale_factor = (self.num_heads * self.num_heads) ** 0.5

    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(self.scale_factor)
        v3 = F.softmax(v2, dim=-1)
        v3 = F.dropout(v3, self.dropout_p)
        v4 = torch.matmul(v3, x3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 5)
x2 = torch.randn(2, 4, 10)
x3 = torch.randn(2, 4, 16)

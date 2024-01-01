
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale_factor = 1 / (dim ** (1 / 4) ** (1 / 2))
        self.dropout_p = 0.01
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim)

    def forward(self, input):
        x = self.query(input)
        y = self.key(input)
        z = self.value(input)
        x = x * self.scale_factor
        y = y * self.scale_factor
        qk = torch.matmul(x, y.transpose(-2, -1))
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(z)
        return output

# Initializing the model
m = Model(dim=3, num_heads=4)

# Inputs to the model
input = torch.randn(1, 5, 3)

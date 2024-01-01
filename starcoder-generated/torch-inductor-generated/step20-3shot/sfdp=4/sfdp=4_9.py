
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    def forward(self, x, x):
        q = x.reshape((-1, self.num_heads, x.shape[-1]))
        k = x.reshape((-1, self.num_heads, x.shape[-1]))
        v = x.reshape((-1, self.num_heads, x.shape[-1]))

        qk = q @ k.transpose(-2, -1)
        a = qk / math.cbrt(q.size(-1))

        return a

# Initializing the model
m = Model(4)

# Inputs to the model
x1 = torch.randn(1, 16, 8)

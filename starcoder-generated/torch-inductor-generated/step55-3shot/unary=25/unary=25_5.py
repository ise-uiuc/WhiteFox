
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x):
        m1 = x.matmul(np.random.randn(2, 3))
        b = m1 > 0
        a = m1 * self.negative_slope
        v = torch.where(b, m1, a)
        return v

# Initializing the model
negative_slope = 0.03
model = Model(negative_slope)

# Initializing an input tensor
x = torch.randn(1, 3)
y = model(x)


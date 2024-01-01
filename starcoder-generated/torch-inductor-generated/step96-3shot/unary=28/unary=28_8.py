
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, False)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = np.clip(m1, self.min_value, None)
        m3 = np.clip(m2, None, self.max_value)
        return m3

# Initializing the model
m = Model(-5, 5)

# Inputs to the model
x1 = torch.randn(25, 16)

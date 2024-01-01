
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = x1 + v1
        m1 = nn.MaxPool1d(2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        v3 = m1(v2)
        m2 = nn.AvgPool1d(2, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        v4 = m2(v3)
        v5 = v4 + v3 # Add the output of the max-pooling operation to the output of the avg-pooling operation
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 8)

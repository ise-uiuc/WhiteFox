
class TestModule_1(torch.nn.Module):
    def __init__(self):
        super(
            TestModule_1,
            self).__init__()
        self.linear = torch.nn.Linear(64, 64, False)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * 0.5
        v3 = v1 + ((v1 * v1) * v1) * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = TestModule_1()

# Inputs to the model
x = torch.randn(1, 64)

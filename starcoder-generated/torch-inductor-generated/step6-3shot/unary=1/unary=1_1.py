
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 16, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 + (v1 * v1 * v1) * 0.044715
        v4 = v3 * 0.7978845608028654
        # TODO: v5 = torch.tanh(v4)
        v5 = torch.sigmoid(v4) + 1
        v6 = v2 * v5
        return v6

m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)

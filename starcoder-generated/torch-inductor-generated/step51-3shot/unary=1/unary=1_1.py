
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 + v1 * v1 * v1 * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = v4 # TanH is hard for ONNX export, so we simply use the original calculation here
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)

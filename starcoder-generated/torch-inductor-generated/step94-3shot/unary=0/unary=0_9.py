
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x56):
        v1 = torch.nn.functional.conv2d(x56, torch.tensor([-1.3367, 1.5913], dtype=torch.float32), None, [5, 3], [3, 2], 1, False, [37, 13], True, True)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x56 = torch.randn(3, 19, 14, 24)


class Model(torch.nn.Module):
    def forward(self, x15):
        v1 = x15 * 0.5
        v2 = x15 * x15
        v3 = v2 * x15
        v4 = v3 * 0.044715
        v5 = x15 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v1 * v8
        return v9
# Inputs to the model
x15 = torch.randn(1, 4, 128, 28)

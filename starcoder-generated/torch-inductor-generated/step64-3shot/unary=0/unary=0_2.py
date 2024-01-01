
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.conv2d(input=x, weight=torch.randn([x.shape[0] * 2, x.shape[1], 2, 2], dtype=torch.float), bias=None, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=x.shape[0])
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
x = torch.randn(1, 3, 128, 28)

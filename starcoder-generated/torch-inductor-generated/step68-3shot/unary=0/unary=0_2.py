
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x44):
        v1 = torch.nn.functional.max_pool2d(x44, 3, padding=0, stride=1, dilation=1, ceil_mode=False)
        v2 = torch.nn.functional.max_pool2d(v1, 8, padding=1, stride=2, dilation=4, ceil_mode=False)
        v3 = torch.nn.functional.leaky_relu(v2, negative_slope=0.010000000000000009, inplace=False)
        v4 = v3 * 0.5389098441162109
        v5 = torch.nn.functional.avg_pool2d(v1, 2, padding=2, stride=6, ceil_mode=False)
        v6 = v5 + 0.02625589760327339
        v7 = v3 * v6
        v8 = torch.nn.functional.relu(v7, inplace=False)
        v9 = v8 + 0.0018362236945724487
        v10 = v4 * v9
        return v10
# Inputs to the model
x44 = torch.randn(1, 256, 4, 29)

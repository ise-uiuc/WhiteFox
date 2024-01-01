
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        negative_slope = -0.80098567
        t1 = F.conv2d(x, torch.randn(6, 6, 3, 3), stride=1, padding=1)
        t2 = t1 > 0
        t3 = t1 * negative_slope
        v2 = t1 + t3
        v1 = list()
        v1.append(F.relu(t1))
        v1.append(torch.sqrt(v2))
        v1.append(-v1[1])
        return v1[2]
# Inputs to the model
x1 = torch.randn(2, 6, 6, 6)

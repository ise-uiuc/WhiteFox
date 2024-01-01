
class Model(torch.nn.Module):
    def __call__(self, x1):
        t1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

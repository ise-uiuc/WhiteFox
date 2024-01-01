
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        n = torch.matmul(v1, v2.transpose(-2, -1))
        k = v1
        v = v2
        s = 0
        for i in range(k.dim()):
            s += k.shape[i]
        d = 1.0 / math.sqrt(s)
        q = torch.nn.functional.softmax(n * d, dim=-1)
        x = torch.matmul(q, v)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

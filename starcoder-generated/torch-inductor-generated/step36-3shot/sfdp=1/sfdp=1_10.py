
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Conv2d(3, 4, 2, stride=1, padding=1)
        self.key = torch.nn.Conv2d(3, 4, 2, stride=1, padding=1)
        self.value = torch.nn.Conv2d(3, 4, 2, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = torch.transpose(self.key(x1), -2, -1)
        v3 = torch.matmul(v1, v2)
        v4 = 1 / (v3.shape[2] * v3.shape[3])
        v5 = torch.nn.functional.dropout(v3 * v4, p=0.7)
        v6 = torch.matmul(v5, self.value(x1))
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_x = torch.nn.Linear(200, 200)
        self.fc_y = torch.nn.Linear(3, 1)
 
    def forward(self, x, y):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = torch.rsqrt(torch.tensor(200, dtype=torch.float))
        v3 = v1 / v2
        v4 = v3.softmax(-1)
        return v4.matmul(self.fc_x(x)) + (self.fc_y(y) * 1.0)

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(30, 200)
y = torch.randn(30, 3)
torch.set_printoptions(precision=6)

# Inference
m(x, y)


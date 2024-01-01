
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        y1 = torch.matmul(x1, x2)
        y2 = torch.linalg.inv(torch.eye(x3.shape[-1]) * (x4 ** 0.5))
        y3 = torch.matmul(y1, torch.transpose(y2))
        y4 = torch.softmax(y3, 1)
        y5 = torch.matmul(x3, y4)
        return y5

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(2, 10, 5)
x2 = torch.randn(2, 5, 10)
x3 = torch.randn(2, 10, 16)
x4 = 10.0

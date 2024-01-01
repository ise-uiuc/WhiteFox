
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        return v2
## Generated input (placeholder)
shape_2 = [1]
x1 = paddle.rand(shape_2)
x2 = paddle.rand(shape_2)
shape_3 = [2]
m = Model()


class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()
    def forward(self, x, y):
        x1 = T.conv_transpose2d(x, [5, 5], 1, stride=[2, 3], padding=[1, 2], output_padding=[1, 2], groups=1)
        x2 = x1 > 0
        x3 = x1 * -3.5345
        x4 = x3
        x5 = x2
        x6 = T.where(x2, x3, x1)
        x7 = T.relu(x6)
        x8 = x1 > 0
        x9 = x1 * 0.83
        x10 = T.where(x2, x3, x3)
        x11 = x10
        x12 = T.relu(x11)
        x13 = x12
        return [x7,]
# Inputs to the model
x = random_tensor(3, 1, 32, 32)
y = random_tensor(3, 5, 22, 14)

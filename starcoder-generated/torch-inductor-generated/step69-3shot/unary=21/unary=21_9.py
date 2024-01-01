
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = ConvBlock()
        self.fc1 = FCBlock()
        self.fc2 = FCBlock()
        self.relu = relu1
        self.tanh = tanh1
    def forward(self, x):
        t1 = self.conv_block(x)
        t2 = self.relu(t1)
        t3 = self.fc1(t2)
        t4 = self.tanh(t3)
        t5 = self.fc2(t4)
        t6 = self.relu(t5)
        t7 = self.fc1(t6)
        t8 = self.tanh(t7)
        t9 = self.fc2(t8)
        t10 = self.relu(t9)
        t11 = self.fc1(t10)
        t12 = self.tanh(t11)
        t13 = (self.fc2(t12))
        return t13
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

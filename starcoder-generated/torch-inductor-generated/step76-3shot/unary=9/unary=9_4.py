
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
    def forward(self, x1):
        t1 = self.relu(self.conv(x1))
        t2 = self.pool(t1)
        t3 = torch.tanh(t2)
        t4 = t3 + 3
        t5 = t4 / 6
        return t5
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

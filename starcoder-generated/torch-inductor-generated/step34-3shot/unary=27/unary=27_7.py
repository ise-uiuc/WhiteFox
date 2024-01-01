
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 11, kernel_size=(2,), stride=(2,), padding=(
            (2,),))
        self.min = min
        self.max = max

    def forward(self, input_tensor=torch.zeros(1, 3, 10, 10)):
        t1 = self.conv(input_tensor)
        t2 = torch.clamp(t1, self.min, self.max)
        t3 = torch.relu(t2)
        return t3
min = 1.1
max = -0.9
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)

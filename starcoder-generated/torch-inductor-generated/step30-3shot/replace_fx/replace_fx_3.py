
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
    def forward(self, x):
        x = torch.rand_like(x)
        return self.conv1(x)
# Inputs to the model
x = torch.randn([1, 3, 244, 244])

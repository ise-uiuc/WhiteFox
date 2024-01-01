
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.rand_like(x)
        return x
# Input to the model
x = torch.randn([1, 3, 244, 244])

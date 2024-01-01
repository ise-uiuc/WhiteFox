
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=1, stride=2)

    def forward(self, x):
        x = torch.rand_like(x)
        x = self.conv1(x)
        x = torch.nn.functional.dropout(x)
        x = torch.rand_like(x)
        x = self.conv2(x)
        x = torch.nn.functional.dropout(x)
        x = torch.rand_like(x)
        return x
# Inputs to the model
x = torch.randn([10, 3, 244, 244])

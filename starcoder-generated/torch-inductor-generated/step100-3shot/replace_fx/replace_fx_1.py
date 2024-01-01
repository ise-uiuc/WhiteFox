
class ExampleModule(torch.nn.Module):
    my_conv2d = torch.nn.Conv2d(16, 32, 3, stride=2)
    my_linear = torch.nn.Linear(123, 456)

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.my_linear  # this is not a real torch.nn.functional.linear call
        self.my_conv2d  # this is not a real torch.nn.functional.conv2d call

    def forward(self, x):
        self.my_conv2d(x)
        self.my_linear(x)
        torch.nn.functional.linear(x, x, True)
        torch.nn.functional.conv2d(x, x, 1)
        return x


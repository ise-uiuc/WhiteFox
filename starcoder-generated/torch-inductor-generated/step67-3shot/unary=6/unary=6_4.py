
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=1)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=1)
        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=1)

    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = torch.cat((t1, t2, t3), dim=0)
        t5 = self.max_pool_1(t4)
        t6 = self.max_pool_2(t4)
        t7 = self.max_pool_3(t4)
        return torch.cat((t5, t6, t7), dim=0).unsqueeze(-1).unsqueeze(0)
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)

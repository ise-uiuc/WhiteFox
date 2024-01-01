
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(3, 8, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(8, 8, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(8, 8, kernel_size=(3, 2), stride=(1, 1), padding=(1, 1))
        self.conv5 = torch.nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1))
    def forward(self, x):
        l1 = {self.conv1}
        l2 = {x, self.conv2}
        l3 = {x, self.conv3}
        v1 = self.conv1(x) # 1
        v2 = self.conv2(l1.pop() + self.conv4(l2.pop() + l3.pop())) # 2
        v3 = v2 - v1 # 3
        v4 = self.conv5(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
# The two ways to generate this model are

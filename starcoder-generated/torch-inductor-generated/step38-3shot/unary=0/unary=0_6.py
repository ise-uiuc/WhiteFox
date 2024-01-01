
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=5, stride=2)
#         self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=5, stride=1)
        self.conv1 = torch.nn.Conv3d(1, 2, kernel_size=[16, 3, 99], stride=[4, 1, 2])
        self.conv2 = torch.nn.Conv2d(2, 4, kernel_size=3, stride=2)
    def forward(self, x2):
        v3 = self.conv1(x2)
        v4 = self.conv2(v3)
        v5 = v4 * 0.5
#         v5 = v4 * 0.707106
#         v6 = v5 #v4 * 0.951198
#         v7 = v6 * 0.507106
        v7 = v5 * 0.707106
        v6 = v5 * float('1e-22') #v4 * 0.951198
        v8 = v6 * 0.507106
#         v9 = v8 * 0.5
        v9 = v8 * 0.707106
        v10 = v9 * 0.707106
        return v10
# Inputs to the model
x2 = torch.randn(1, 1, 24, 487, 799)

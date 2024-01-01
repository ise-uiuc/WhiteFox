
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
#         self.conv2 = torch.nn.Conv2d(8, 32, 1, stride=1, padding=1)
#         self.bn1 = torch.nn.BatchNorm2d(32)
#         self.bn2 = torch.nn.BatchNorm2d(32)
#     def forward(self, x):
#         v1 = self.conv1(x)
#         v2 = self.conv2(v1)
#         v3 = self.bn1(v2)
#         v4 = self.bn2(v2)
#         v5 = v3 + v4
#         return v5
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)


# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
#         self.conv_2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
#     def forward(self, x1):
#         v1 = self.conv_1(x1)
#         v2 = F.sigmoid(v1)
#         v3 = F.sigmoid(v2)
#         v4 = v1 * v2
#         v5 = v1 + v3
#         v6 = v4 + v5
#         return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

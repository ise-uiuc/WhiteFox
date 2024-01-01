
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv2d = torch.nn.ConvTranspose2d(2, 69, 13, stride=3, padding=9, dilation=6)
#     def forward(self, x1):
#         x2 = self.conv2d(x1)
#         z6 = x2 > 0
#         z7 = x2 * -0.11985
#         z8 = torch.where(z6, x2, z7)
#         return z8
#
# # Inputs to the model
# x1 = torch.randn(2, 2, 18, 18)
# 
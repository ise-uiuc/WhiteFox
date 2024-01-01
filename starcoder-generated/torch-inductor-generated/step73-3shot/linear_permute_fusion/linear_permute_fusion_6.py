
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(2, 2)
#     def forward(self, x2):
#         v1 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
#         v2 = v1.permute(0, 2, 1)
#         return v2.flatten(0, 1)
# Inputs to the model
# x2 = torch.randn(1, 2, 2)

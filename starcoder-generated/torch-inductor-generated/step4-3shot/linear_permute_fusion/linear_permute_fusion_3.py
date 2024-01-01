
# PyTorch does not allow to input 4D tensors with more than 4 dimensions into linear transformation. 
# Comment out or delete this model if your torch.nn.functional.linear can accept the tensors with 4 or more dimensions.
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(2, 3)
#     def forward(self, x1):
#         v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
#         v2 = v1.permute(0, 2, 1, 3)
#         return v2
# # Inputs to the model
# x1 = torch.randn(1, 1, 2, 2)
# 

class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         v1 = torch.Tensor([
#             [-1, -1, -1],
#             [-1, 8, -1],
#             [-1, -1, -1]
#         ])
#         v2 = torch.Tensor([
#             [0, 1, 0],
#             [1, -4, 1],
#             [0, 1, 0]
#         ])
        v1 = torch.tensor([0.22, 0.26])
        v2 = torch.tensor([[0.2504, 0.2504],
                      [-0.0098, 0.1895]])
        self.convolution2d = torch.nn.Conv2d(2, 1, (3, 3), stride=1, padding=1, bias=False)
        self.softmax = torch.nn.Softmax(1)
        self.weights = torch.nn.Parameter(torch.stack([v1[None, :, None], v2[None, :, None]], dim=1), requires_grad=True)
    def forward(self, x):
        v1 = self.convolution2d(x * self.weights)
        v2 = self.softmax(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
tensor = torch.randn(1, 2, 16, 16)

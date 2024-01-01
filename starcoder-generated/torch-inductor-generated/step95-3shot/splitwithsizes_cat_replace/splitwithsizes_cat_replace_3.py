
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = torch.nn.Parameter(torch.randn((9, 9, 3, 3)), requires_grad=True)
    def forward(self, v1):
        split_tensors = torch.split(v1, [3, 3, 3], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [3, 3, 3], dim=2))
# Inputs to the model
x1 = torch.randn(1, 1, 27, 3)

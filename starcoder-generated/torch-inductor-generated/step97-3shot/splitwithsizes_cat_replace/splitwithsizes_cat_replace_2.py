
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.ModuleList([torch.nn.Conv1d(4, 8, 3, 1, 1, bias=True), torch.nn.PReLU(), torch.nn.Conv1d(8, 16, 3, 1, 1), torch.nn.ReLU()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

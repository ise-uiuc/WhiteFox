
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv1d(1, 1, kernel_size=(2)), torch.nn.BatchNorm1d(1)])
    def forward(self, x):
        split_tensors = torch.split(x, [2], dim=0)
        x1 = torch.cat([split_tensors[0], split_tensors[1], split_tensors[3], split_tensors[4]], dim=0)
        return x1
# Inputs to the model
x = torch.randn(8, 1)

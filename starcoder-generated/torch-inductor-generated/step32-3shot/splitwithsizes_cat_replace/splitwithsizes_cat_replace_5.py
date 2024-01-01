
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.ReLU()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 64, 3, 1, 1), self.__0, torch.nn.Conv2d(64, 64, 3, 1, 1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, self.__0(v1), torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

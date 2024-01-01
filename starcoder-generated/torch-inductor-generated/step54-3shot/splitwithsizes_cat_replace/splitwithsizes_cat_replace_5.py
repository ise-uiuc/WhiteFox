
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        list0_0 = [torch.nn.BatchNorm2d(3)]
        list0_1 = [torch.nn.ReLU()]
        list1_0 = [torch.nn.BatchNorm2d(3)]
        list1_1 = [torch.nn.ReLU()]
        list2_0 = [torch.nn.BatchNorm2d(3)]
        list2_1 = [torch.nn.ReLU()]
        self.features = torch.nn.ModuleList([torch.nn.Sequential(*list0_0, *list0_1), torch.nn.Sequential(*list1_0, *list1_1), torch.nn.Sequential(*list2_0, *list2_1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Block is just for this test, not used in the original model generated
        block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor1 = torch.cat(split_tensors[:len(split_tensors) - 1], dim=1)
        concatenated_tensor2 = torch.cat(split_tensors[1: len(split_tensors)], dim=1)
        return (concatenated_tensor2, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([Model1()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

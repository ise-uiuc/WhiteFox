
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.MaxPool2d(kernel_size=2), torch.nn.BatchNorm2d(32), torch.nn.Linear(32, 64), torch.nn.MaxPool2d(kernel_size=2), torch.nn.BatchNorm1d(64), torch.nn.Softmax(dim=-1)])
    def forward(self, input_tensor):
        split_tensors = torch.split(self.layers[0](input_tensor), [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([self.layers[1](split_tensors[0]), self.layers[2](split_tensors[1]), self.layers[3](split_tensors[2])], dim=1)
        return (concatenated_tensor, torch.cat([split_tensors[0], split_tensors[1], split_tensors[2]], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

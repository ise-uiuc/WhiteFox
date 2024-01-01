
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 2, 3), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.ReLU(inplace=False)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(_forward_relu(split_tensors), dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))

    def _forward_relu(self, split_tensors):
        xs = []
        for x in split_tensors:
            x = torch.sigmoid(x)
            x = torch.tanh(x)
            xs.append(x)
        return xs
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

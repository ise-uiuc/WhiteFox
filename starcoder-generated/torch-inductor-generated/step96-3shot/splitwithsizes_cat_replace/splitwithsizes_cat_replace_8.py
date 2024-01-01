
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.Dropout2d(0.0001)
        self.features2 = torch.nn.ReLU(inplace=False)
        self.features3 = torch.nn.Linear(2, 4)
        self.features4 = torch.nn.Sigmoid()
    def forward(self, a, b):
        split_tensor = torch.split(a, [1, 1], dim=1)
        concatenated_tensor3 = torch.cat([split_tensor[0], split_tensor[1], b], dim=1)
        split_tensor_1 = torch.split(concatenated_tensor3, [2, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensor_1, dim=1)
        return (concatenated_tensor, torch.split(a, [1, 1], dim=1))
# Inputs to the model
x = torch.randn(1, 2)
x1 = torch.randn(1, 3, 224, 224)

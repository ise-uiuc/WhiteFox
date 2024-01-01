
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features0 = torch.nn.Conv2d(3, 32, 3, 2, 2, bias=False)
        self.features1 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.activation1 = torch.nn.ReLU()
        self.features2 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.activation2 = torch.nn.ReLU()
        self.join_tensor = torch.nn.Sequential()
        self.features7 = torch.nn.Conv2d(32, 32, 3, 2, 2, bias=False)
        self.features8 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.activation3 = torch.nn.ReLU()
        self.features9 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.activation4 = torch.nn.ReLU()
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

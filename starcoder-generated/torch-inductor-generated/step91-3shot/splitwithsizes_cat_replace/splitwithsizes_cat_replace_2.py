
class Layer0(torch.nn.Module):
    def __init__(self, inp, out):
        super(Layer0, self).__init__()
        self.block = Block()
    def forward(self, inputs):
        split_tensors = torch.split(inputs, [1, 1, 1], 1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return self.block(concatenated_tensor)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = Layer0(3, 32)
        self.extra = torch.nn.ReLU()
    def forward(self, inputs):
        split_tensors = torch.split(inputs, 10, 1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        split_tensor_list = torch.split(x1, 2, 1)
        tensors = [split_tensor_list[i] * 0.5 for i in range(1)]
        concatenated_tensor = torch.cat(tensors, 1)
        return concatenated_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 2, 2)

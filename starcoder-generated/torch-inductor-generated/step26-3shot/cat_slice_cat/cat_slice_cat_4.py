
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *input_tensors):
        tensors = list(input_tensors)
        tensor1 = torch.cat(tensors, dim=1)
        tensor11 = tensor1[:, 0:9223372036854775807]
        tensor2 = tensor11[:, 0:len(tensors)]
        tensor3 = torch.cat([tensor1, tensor2], dim=1)
        return tensor3

# Initializing the model
m = Model()

# Inputs to the model
inputs = [torch.randn(1, 4) for _ in range(3)]

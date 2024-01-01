
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        split_tensors = torch.split(x1, [5, 6, 3], 1)
        v1 = torch.tensor([0.0625, 0.125, 0.25, 0.375, 0.5, 0.625])
        concatenated_tensor = v1 * torch.cat([split_tensors[1], v1 * split_tensors[0], split_tensors[2], split_tensors[1]], 1)
        return True

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 15, 23, 5)

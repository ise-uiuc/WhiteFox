
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.split(x, 2, dim = 0) #split in 0-dimension
        x1 = torch.squeeze(x1) #remove 0-dimension
        tensor1, tensor2 = torch.split(x, 3, dim = 1)
        return tensor1
# Inputs to the model
x = torch.randn(2, 3, 4)

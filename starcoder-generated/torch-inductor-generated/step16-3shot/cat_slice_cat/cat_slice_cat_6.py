
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, size, tensors1, tensors2, tensors3):
        t1 = torch.cat(tensors1, tensors2, tensors3, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        t4 = torch.cat([t1, t3])
        return t4

# Initializing the model
m = Model()

# Inputs to the model
size = 9223372036854775807
tensor1 = torch.randn(1, 4, 5, 5)
tensor2 = torch.randn(1, 4, 5, 6)
tensor3 = torch.randn(1, 4, 5, 7)

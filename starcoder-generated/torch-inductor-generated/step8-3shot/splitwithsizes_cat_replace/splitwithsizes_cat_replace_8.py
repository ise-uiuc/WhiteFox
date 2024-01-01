
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, input):
        split_tensors = torch.split(input, [1, 1, 3, 1, 2], 1)
        concatenated_tensor = torch.cat(split_tensors, 1)

        v1 = torch.split(concatenated_tensor, [1, 2], 1)
        c = torch.cat(v1, 1)

        return c


# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 13, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(160, 1, 3)
        # TODO: initialize weights and bias such that the conv layer produces the desired output pattern
        # Refer to readme.md for instructions on how to initialize weights and bias
        self.conv_1.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(64, 160, 3, 3),(0.01), (0.1)))
        self.conv_1.bias = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(64, 160, 3, 3), 0.01,0.1))
        self.conv_2 = torch.nn.Conv2d(64, 1, 3)
        # TODO: initialize weights and bias such that the conv layer produces the desired output pattern
        # Refer to readme.md for instructions on how to initialize weights and bias
        self.conv_2.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(160, 64, 3, 3),(0.01), (0.1)))
        self.conv_2.bias = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(160, 64, 3, 3), 0.01,0.1))

    def forward(self, x1):
        v1 = self.conv_1(x1) # apply conv layer "conv_1"
        v2 = F.sigmoid(v1) # apply "sigmoid" activation function
        v3 = v2 * v1 # multiply by the output of "conv_1"
        v3 = v3.sum(dim = 1)
        v4 = self.conv_2(v3) # apply conv layer "conv_2"
        return v4

# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)

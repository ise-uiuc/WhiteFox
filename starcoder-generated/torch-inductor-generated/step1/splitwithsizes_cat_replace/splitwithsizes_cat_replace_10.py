
class Model(torch.nn.Module):
    def __init__(self, input_):
        super().__init__()
        dim0, dim1 = torch.split(torch.split(input_, 2, 0)[0], input_.size(1), 2).size()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.split1 = torch.split_with_sizes(input_, [dim0], 0)
        self.split2 = torch.split_with_sizes(input_, [dim1], 1)
 
    def forward(self, x):
        v0 = x * 10
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.cat([self.split1[1], v0, self.split1[0], v2], 2)
        v4 = torch.cat([self.split2[1], v0, self.split2[0], v3], 1)
        return v4

# Initializing the model
m = Model(x)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

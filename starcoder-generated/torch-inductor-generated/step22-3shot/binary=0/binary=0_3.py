
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            ["conv", torch.nn.Conv2d(3, 8, 1, stride=1, padding=1), 1],
        ]
    def forward(self, x1, other=1, padding1=None):
        for k in range(len(self.layers)):
            [op_name, op, stride] = self.layers[k]
            if   op_name == "pool":
                x1 = F.avg_pool2d(x1, op.kernel_size, stride=op.stride, padding=op.padding)
            elif op_name == "conv":
                x1 = op(x1)
            elif op_name == "bn":
                x1 = op(x1, self.training)
            elif op_name == "relu": # TODO: Use inplace ReLU if possible.
                x1 = F.relu(x1)
            elif op_name == "fc":
                x1 = F.linear(x1.flatten(1), op.weight, op.bias)
            elif op_name == "add":
                x1 = x1 + op
            if padding1 == None:
                padding1 = torch.randn(x1.shape)
            x1 = x1 + other
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

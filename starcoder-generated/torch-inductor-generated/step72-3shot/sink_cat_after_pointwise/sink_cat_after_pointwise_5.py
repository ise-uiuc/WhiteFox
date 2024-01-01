
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape_list = list(x.shape)
        for j in range(4):
            shape_list[j] = -1
        x = math_ops.relu(x)
        x = x.view(*shape_list)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

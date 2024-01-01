
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        concat = torch.cat([x, x], dim=1).squeeze()
        y = concat.permute([1, 0, 2]).contiguous()
        x = y.view([concat.shape[1], -1]) if concat.shape[1] > 10 else y.view(concat.shape[1], -1).contiguous()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

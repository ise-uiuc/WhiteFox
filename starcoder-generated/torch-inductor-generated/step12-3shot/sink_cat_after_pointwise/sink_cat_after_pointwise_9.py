
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        tensor_list = [x, x, x]
        x = torch.cat(tensor_list, dim=1)
        x = x.view(x.shape[0], 6) if x.shape == (1, 18) else x.view(x.shape[0], 6)
        x = x.tanh() if x.shape == (1, 6) else x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

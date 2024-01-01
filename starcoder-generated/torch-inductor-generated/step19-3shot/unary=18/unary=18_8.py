
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forwadi(self, x):
        x_new = x.view(-1, 8, 2, int(x.size(2)/2), int(x.size(3)/2))
        x_new = x_new.view(-1, 8, 1, int(x.size(2)/2), int(x.size(3)/2))
        return x_new

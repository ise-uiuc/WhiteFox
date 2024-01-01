
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, 0, 0), torch.nn.MaxPool2d(3, 4, 1, 1), torch.nn.MaxPool2d(2, 2, 0, 0))
        self.transpose = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 32, 3, 1, 1), torch.nn.ConvTranspose2d(32, 3, 3, 1, 1))
        self.view = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
    def forward(self, x1):
        max_pool_tensor = self.pool(x1)
        transpose_tensor = self.transpose(max_pool_tensor)
        view_tensor = self.view(max_pool_tensor)
        view_tensor = view_tensor.view([3, 60, -1, 7])
        view_tensor = view_tensor.permute([0, 2, 3, 1])
        return torch.cat([transpose_tensor, view_tensor], 3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

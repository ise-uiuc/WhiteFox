
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 75)
        self.linear_2 = torch.nn.Linear(75, 75)
        self.linear_3 = torch.nn.Linear(75, 3)
    def forward(self, x):
        batch_dim, feature_dim = x.shape
        x = self.linear_1(x)
        x = self.linear_2(x)
        result = self.linear_3(x)
        return result.view(batch_dim, 1, 3)

x = torch.randn(3,2)

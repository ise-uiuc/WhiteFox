     
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.conv2d(x, torch.rand(3, 3, 3, 3), stride=1)
        x = F.batch_norm(x, num_features=3)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)

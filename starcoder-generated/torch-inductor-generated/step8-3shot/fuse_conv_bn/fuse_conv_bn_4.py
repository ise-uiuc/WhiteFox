
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.conv2d(x1, torch.randn(2,2,3,3), None, stride=2, padding=0)
        x2 = torch.nn.functional.batch_norm(x2, torch.tensor([0.3, 0.5]), torch.tensor([1, 2]), torch.randn(2), torch.randn(2))
        x3 = torch.nn.functional.relu(x2, inplace=True)
        y2 = torch.nn.functional.batch_norm(x3, torch.tensor([0.1, 0.7]), torch.tensor([3, 4]), torch.randn(2), torch.randn(2))
        return y2
# Inputs to the model
x1 = torch.randn(2, 2, 5, 5)

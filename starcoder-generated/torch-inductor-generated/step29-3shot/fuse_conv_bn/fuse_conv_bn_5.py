
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
   self.layer = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(2))
    def forward(self, input_tensor):
        return self.layer(input_tensor)
# Input to the model
input_tensor = torch.randn(1, 2, 5, 5)

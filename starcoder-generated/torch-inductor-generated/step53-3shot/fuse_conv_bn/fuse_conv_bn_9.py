
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
    def forward(self, input_tensor):
        fc_result = torch.nn.functional.conv2d(input_tensor, self.fc1.weight, self.fc1.bias)
        self.bn = torch.nn.BatchNorm2d(3)
        return fc_result
# Inputs to the model
input_tensor = torch.randn(1, 3, 3, 3)

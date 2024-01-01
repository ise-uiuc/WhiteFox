
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = self.conv2(input)
        v3 = self.bn1(v2)
        v4 = self.bn2(v1 + v2)
        return v3
# Inputs to the model
input = torch.randn(1, 3, 64, 64)  # (N, C, Hieght, Width)
torch.onnx.export(nn.Sequential(Model()), input, "model.onnx", output_names=["output"], opset_version=12, verbose=False)

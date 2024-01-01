
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(10):
            conv = torch.nn.Conv2d(16, 16, 3, padding=1)
            bn = torch.nn.BatchNorm2d(16)
            relu = torch.nn.ReLU()
            l = torch.nn.Sequential(* [ conv, bn, relu ])
            self.conv_layers.append(l)
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
# Inputs to the model
tensor = torch.randn(1, 1, 33, 33)

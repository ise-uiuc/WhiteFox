
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained = False)
        self.resnet.eval()
    def forward(self, x):
        v1 = self.resnet(x)
        v2 = v1.unsqueeze(-1).expand(v1.size(0), v1.size(1), 224)
        v3 = v2.permute(1, 0, 2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

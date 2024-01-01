
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
    def forward(self, s1):
        v1 = s1
        v2 = v1.view(-1, 512, s1.shape[2], s1.shape[3])
        v3 = self.conv(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
s1 = torch.randn(3, 512, 8, 8)
# Model Ends
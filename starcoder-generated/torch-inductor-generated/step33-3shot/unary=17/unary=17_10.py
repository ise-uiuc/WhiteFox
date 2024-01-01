
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1=torch.nn.ConvTranspose2d(3, 64, (1, 1), stride=(1, 1), padding=(0, 0)
        self.conv1_2=torch.nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2), padding=(0, 0))
        self.conv1_3=torch.nn.ConvTranspose2d(64, 512, (4, 4), stride=(4, 4), padding=(0, 0))
        self.flatten=torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_ = torch.nn.Linear(2048,2)
    def forward(self, x1):
        v1 = self.conv1_1(x1)
        v2 = F.tanh(v1)
        v3 = self.conv1_2(v2)
        v4 = F.tanh(v3)
        v5 = self.conv1_3(v4)
        v6 = F.tanh(v5)
        v7 = self.flatten(v6)
        v8 = self.linear_(v7)
        return v8
# Inputs to the model
x1, _ = torch.utils.model_zoo.load_url(model_urls['vgg16_bn'])

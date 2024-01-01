
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 192
        self.embed_size = 512
    def forward(self, input):
        return torch.softmax(input, dim=-1)
# Inputs to the model
input = torch.randn(1, 192, 512)


class Model(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout2d(p)
    def forward(self, img):
        img = self.dropout(img)
        return img
# Inputs to the model
img = torch.randn(3, 12, 12)

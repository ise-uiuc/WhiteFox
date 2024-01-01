
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mask, values, keys, queries):
        
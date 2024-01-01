
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.init.kaiming_uniform_(torch.empty(5, 4, 3, 3, dtype=torch.float32), a=math.sqrt(5), mode = 'fan_in', nonlinearity='relu')
        
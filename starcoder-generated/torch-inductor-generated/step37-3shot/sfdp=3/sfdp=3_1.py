
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, input):
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)
        

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 33).to('cuda:1')
        self.conv = torch.nn.Conv2d(2, 2, 1).to('cuda:1')
    def forward(self, x) -> torch.Tensor:
        x = self.fc(x).detach()
        if (x.shape!= (3, 33)):
            x = x.view(3, 33)
        x = self.conv(x)
        print('Input shape:', x.shape)
        return x
# Input to the model
x = torch.randn(1, 2, requires_grad=True).to('cuda:1')
y = torch.randint(3, (3, 2), dtype=torch.long, requires_grad=True).to('cuda:1')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64 * 2 * 2, 3),
            torch.nn.Softmax()
        )
        self.model = model
    def forward(self, x1):
        x = torch.unsqueeze(x1, 0)
        # x = torch.transpose(x, dim0=0, dim1=1)
        return torch.squeeze(self.model(x), dim=0)

x1 = torch.randn(224, 224, 3)

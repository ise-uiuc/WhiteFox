
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(20, 40, 2, stride=2, padding=2)
    def forward(self, x20):
        w1 = self.conv_t(x20)
        w2 = w1 > 0
        w3 = w1 * -0.4511
        w4 = torch.where(w2, w1, w3)
        return torch.nn.functional.batch_norm(w4, weight=torch.randn(128), bias=torch.randn(128), running_mean=torch.randn(128), running_var=torch.randn(128), training=False, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Inputs to the model
x20 = torch.randn(5, 20, 35, 100)

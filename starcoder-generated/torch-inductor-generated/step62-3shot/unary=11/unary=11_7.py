
class Model(torch.nn.Module):
    def __init__(self, t=3):
        super().__init__()
        # This pattern works for the following modules:
        self.modules = [
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,

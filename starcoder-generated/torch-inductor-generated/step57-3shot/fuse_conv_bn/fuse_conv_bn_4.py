
class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1)
            self.layer1 = torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, False, torch.nn.Conv2d(16, 16, 2, 1, 0, 1, 1, 1, Fal, dtype=torch.float32, device=cpu), forward)
# Inputs to the model
x = torch.randn(1, 16, 16, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(  
            torch.nn.Conv2d(3, 8, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),          
            torch.nn.Conv2d(8, 8, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(8),    
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),                 
            torch.nn.Conv2d(8, 16, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),                     
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(16, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),                                
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
    def forward(self, x):
        return self.blocks(x)
# Inputs to the model
x = torch.randn(2, 3, 24, 24)

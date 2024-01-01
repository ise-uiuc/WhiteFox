
class Model(torch.nn.Module):
    def __init__(self, size, padding=4):
        super().__init__()
        self.size = size
        self.padding = padding
 
    def get_sliced_tensors(self, x1, x2, x3):
        b, c, h, w = x1.shape
        pad_h = h + self.padding * 2
        pad_w = w + self.padding * 2
        
        x1 = F.pad(x1, (self.padding, pad_w - w - self.padding, self.padding, pad_h - h - self.padding)) # padding
        x2 = F.pad(x2, (self.padding, pad_w - w - self.padding, self.padding, pad_h - h - self.padding)) # padding
        x3 = F.pad(x3, (self.padding, pad_w - w - self.padding, self.padding, pad_h - h - self.padding)) # padding

        slice_h = h - self.size
        slice_w = w - self.size
        pad_h = self.padding
        pad_w = self.padding
        
        x1 = x1[:, :, pad_h:pad_h + slice_h, pad_w:pad_w + slice_w] # slicing
        x2 = x2[:, :, pad_h:pad_h + slice_h, pad_w:pad_w + slice_w] # slicing
        x3 = x3[:, :, pad_h:pad_h + slice_h, pad_w:pad_w + slice_w] # slicing
        return x1, x2, x3
    
    def forward(self, x1, x2, x3):
        x1, x2, x3 = self.get_sliced_tensors(x1, x2, x3)
        return torch.cat([x1, x2, x3], dim=1) # concatenate input tensors along dimension 1

# Initializing the model
m = Model(17)

# Input to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
x3 = torch.randn(2, 3, 64, 64)

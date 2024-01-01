
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = np.random.uniform(low=0.0, high=0.1)
 
    def forward(self, __x1__, __x2__, __x3__):
        qk = torch.matmul(__x1__, __x2__.transpose(-2, -1)) 
        
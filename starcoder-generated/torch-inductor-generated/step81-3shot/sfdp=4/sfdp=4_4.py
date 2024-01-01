
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()

        self.key_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(inplace=True),
            )
        self.query_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(inplace=True),
            )
        self.value_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out

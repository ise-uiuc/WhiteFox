. The model is taken from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
class ContentLoss(torch.nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        #.view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
class StyleTransfer(torch.nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        # content loss
        # 3 is the number of channels of the output of the convolution
        # ReLU layers in the network.
        content_layers_default=['conv_4']
        style_layers_default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_layers = content_layers_default
        self.style_layers = style_layers_default
# Input images
        # if you want to use a pre-trained model,
        # change the model below for a new one.
        self.cnn = models.vgg19(pretrained=True).features.eval()
        # normalization module
        self.normalization = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# style loss
        # just in order to have an iterable access to or list of content/syle
        # losses
        self.style_losses = []
        self.content_losses = []
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalization)
# add content loss
# assuming that cnn is a nn.Sequential, the inputs for content and style
# losses can be either the output from a layer (after activation) or the
# input image
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name ='relu_{}'.format(i)
                # inplace version is much faster
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)
            # add style loss
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)
# just in order to have an iterable access to content/syle
        # losses
        self.model = model
# gram matrix and loss
# gram_matrix function from pytorch gram matrix tutorial
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        #.view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
class StyleTransfer(torch.nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        # convolutions
        # number of convolution filters for the first layer
        self.cn1 = 16
        # number of max-pooling layers
        self.mps = 5
        # number of convolution filters for the second layer
        self.cn2 = 32
        # number of upsampling layers
        self.us = 3

# Input images
        # if you want to use a pre-trained model,
        # change the model below for a new one
        self.cnn = models.vgg19(pretrained=True).features.eval()
        # normalizing
        self.normalization = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# convolutions
        self.convs = nn.Sequential(
        # normalize
        self.normalization,
        # conv1_1
        nn.Conv2d(3, self.cn1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        # conv1_2
        nn.Conv2d(self.cn1, self.cn1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
# max-pooling layers
        for i in range(self.mps):
            self.convs.add_module("pool_{}".format(i + 1), nn.Conv2d(self.cn1, self.cn1, kernel_size=3, padding=1))
        self.convs.add_module("conv_{}".format(self.mps + 1), nn.Conv2d(self.cn1, self.cn2, kernel_size=3, padding=1))
        self.convs.add_module("relu_{}".format(self.mps + 2), nn.ReLU(inplace=False))

# upsampling layers
        self.deconvs = nn.Sequential(
        nn.ConvTranspose2d(self.cn2, self.cn2, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=False),
        nn.ConvTranspose2d(self.cn2, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=False),
        )
class NeuralStyleTransfer(torch.nn.Module):
    def __init__(self, content_img, style_img):
        super(NeuralStyleTransfer, self).__init__()
        # get content and style image

content_img = cv2.imread('content.jpg')
style_img = cv2.imread('style.jpg')

content_img = torch.from_numpy(content_img).float().unsqueeze(0).permute(0, 3, 1, 2)
style_img = torch.from_numpy(style_img.swapaxes(1, 2).swapaxes(0, 1)).float().unsqueeze(0).permute(0, 3, 1, 2)

# define style transfer model

# put the model on cuda if cuda can be detected on the devies
# and set up the default device if cuda can't be detected on the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on device: {}'.format(device))

# initialize style transfer network with images

        self.model = StyleTransfer().to(device)
        self.style_img = torch.autograd.Variable(style_img, False).to(device)
        self.content_img = torch.autograd.Variable(content_img, True).to(device)

    def forward(self):
        
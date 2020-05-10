<h1 align = "center"> DCGAN and Wasserstein GAN Implementation </h1>

Below I will be implementing the DCGAN (read more [here](https://arxiv.org/abs/1511.06434) and thereafter, I will be implementing the Wasserstein GAN (read more [here](https://arxiv.org/pdf/1704.00028.pdf)).
First we define the generator and discriminator class for use in the DCGAN and the (improved) Wasserstein GAN:

```
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d( args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False)
        )
 ```
 We can re-use the same generator/discriminator classes for use in the improved Wasserstein GAN, where the main difference is that we use calculate the gradient penalty.
 > **Note:** Adding the gradient penalty will make the model train with far longer times:
 
 ```
 def calc_gradient_penalty(real_imgs, fake_imgs):
    alpha = torch.rand(real_imgs.size(0), 1)
    alpha = alpha.expand(real_imgs.size(0), 3*64*64).contiguous().view(real_imgs.size(0), 3, 64, 64).to(device)
    interpolates = alpha * real_imgs.detach() + ((1 - alpha) * fake_imgs.detach()).to(device)
    interpolates.requires_grad_(True)
    D_inter, inter_logits = netD(interpolates)
    gradients = torch.autograd.grad(outputs=inter_logits, inputs=interpolates, grad_outputs=torch.ones(inter_logits.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lamb
    return gradient_penalty
```

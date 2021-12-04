import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from lbfgs import lbfgs


class InverseBlock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(InverseBlock, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_chn, out_chn, 1, 1, bias=False),
            nn.BatchNorm2d(out_chn)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_chn, out_chn * 2, 1, 1, bias=False),
            nn.BatchNorm2d(out_chn * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_chn * 2, out_chn * 2, 3, 1, 1, bias=False, groups=16),
            nn.BatchNorm2d(out_chn * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_chn * 2, out_chn * 2, 1, 1, bias=False),
            nn.BatchNorm2d(out_chn * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(out_chn * 2, out_chn * 2, 4, 2, 1, bias=False, groups=16),
            nn.BatchNorm2d(out_chn * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_chn * 2, out_chn, 1, 1, bias=False),
            nn.BatchNorm2d(out_chn)
        )

        self.output = nn.Sequential(
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_chn, out_chn, 1, 1)
        )

    def forward(self, x):
        return self.output(self.upsample(x) + self.residual(x))


class InvertVgg(nn.Module):
    def __init__(self):
        super(InvertVgg, self).__init__()

        self.stage5 = InverseBlock(512, 512)
        self.stage4 = InverseBlock(512, 256)
        self.stage3 = InverseBlock(256, 128)
        self.stage2 = InverseBlock(128, 64)

    def forward(self, feats):
        return {i-1:getattr(self, f"stage{i}")(feats[i]) for i in range(2, 6)}


class Vgg19:
    def __init__(self, device):
        self.device = device
        url = "https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth"
        vgg19_model = models.vgg19(pretrained=False).eval()
        vgg19_model.load_state_dict(model_zoo.load_url(url), strict=False)

        vgg19_model = vgg19_model.to(device)
        for name, module in vgg19_model.named_modules():
            if hasattr(module, "inplace"):
                module.inplace = False

        self.stage1 = nn.Sequential(*[vgg19_model.features[i] for i in range(1)])
        self.stage2 = nn.Sequential(*[vgg19_model.features[i] for i in range(1, 6)])
        self.stage3 = nn.Sequential(*[vgg19_model.features[i] for i in range(6, 11)])
        self.stage4 = nn.Sequential(*[vgg19_model.features[i] for i in range(11, 20)])
        self.stage5 = nn.Sequential(*[vgg19_model.features[i] for i in range(20, 29)])

        self.mean_ = torch.tensor((103.939, 116.779, 123.68)).to(device).view(1, 3, 1, 1)

        if device == "cpu":
            invert_model = InvertVgg()
            invert_model.load_state_dict(torch.load("invert_vgg.ckpt"))
            invert_model = invert_model.eval().to(device)
            for i in range(2, 6):
                setattr(self, f"inv_stage{i}", getattr(invert_model, f"stage{i}"))

    def get_features(self, img):
        img = img.float().to(self.device)
        if img.dim() == 3: img = img.unsqueeze(0)
        img -= self.mean_

        output = [img]
        with torch.no_grad():
            for i in range(1, 6):
                output.append(getattr(self, f"stage{i}")(output[-1]))
        return output

    def get_deconvoluted_feat(self, feat, stage, init=None, lr=10, iters=13, display=False):

        source = init.float().to(self.device)
        target = feat.float().to(self.device)
        source_size = source.size()

        if self.device == "cpu":
            with torch.no_grad():
                output = getattr(self, f"inv_stage{stage}")(target)
                loss = (getattr(self, f"stage{stage}")(output) - target).pow(2).mean()
                print(f"loss: {loss:.2f}")
            return output.float()

        # ================
        def loss_func(x):
            x = x.view(source_size)
            x = getattr(self, f"stage{stage-1}")(x)
            output = getattr(self, f"stage{stage}")(x)
            squared_error = (output - target).pow(2).mean()
            return squared_error

        def closure(x):
            x = x.clone().requires_grad_(True)
            loss = loss_func(x)
            loss.backward()
            grad = x.grad.view(-1)
            return loss.item(), grad
        # ================

        init_loss = loss_func(source).item()
        last_loss = init_loss
        source = source.contiguous().view(-1)
        histSize = 4
        for idx in range(2):
            source_out, stat = lbfgs(closure, source, maxIter=iters, histSize=histSize, lr=lr, display=display)
            if stat in ["LBFGS REACH MAX ITER", "LBFGS BELOW GRADIENT EPS"]:
                source = source_out
                break
            new_loss = loss_func(source_out)
            if not torch.isnan(new_loss).item() and new_loss.item() < last_loss:
                source = source_out
            histSize = 10
            lr = lr / 10

        end_loss = loss_func(source).item()
        print('\tstate:' + stat)
        print('\tend_loss/init_loss: {:.2f}/{:.2f}'.format(end_loss, init_loss))

        source = source.view(source_size)
        with torch.no_grad():
            out = getattr(self, f"stage{stage-1}")(source).float()
        return out

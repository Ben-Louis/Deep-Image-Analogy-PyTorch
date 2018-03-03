"""
This code is modified from harveyslash's work (https://github.com/harveyslash/Deep-Image-Analogy-PyTorch)
"""


import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from lbfgs import lbfgs
import copy


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x):
        list = []
        for module in self._modules:
            x = self._modules[module](x)
            list.append(x)
        return list


class VGG19:
    def __init__(self, use_cuda=True):

        url = "https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth"
        vgg19_model = models.vgg19(pretrained=False)
        vgg19_model.load_state_dict(model_zoo.load_url(url), strict=False)
        self.cnn_temp = vgg19_model.features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1
        self.use_cuda = use_cuda

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.MaxPool2d((2, 2), ceil_mode=True))  # ***

            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter) + "__" + str(i)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

        if use_cuda:
            self.model.cuda()
        self.mean_ = (103.939, 116.779, 123.68)

    def forward_subnet(self, input_tensor, start_layer, end_layer):
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                input_tensor = layer(input_tensor)
        return input_tensor

    def get_features(self, img_tensor, layers):
        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        # assert torch.max(img_tensor)<=1.0 and torch.min(img_tensor)>=0.0, 'inccorect range of tensor'
        for chn in range(3):
            img_tensor[:, chn, :, :] -= self.mean_[chn]

        img_tensor = Variable(img_tensor)
        # img_tensor = F.relu(img_tensor)
        features_raw = self.model(img_tensor)
        features = []
        for i, f in enumerate(features_raw):
            if (i) in layers:
                features.append(f.data)
        features.reverse()
        features.append(img_tensor.data)

        sizes = [f.size() for f in features]
        return features, sizes

    def get_deconvoluted_feat(self, feat, curr_layer, init=None, lr=10, iters=13, display=False):

        blob_layers = [29, 20, 11, 6, 1, -1]
        end_layer = blob_layers[curr_layer]
        mid_layer = blob_layers[curr_layer + 1]
        start_layer = blob_layers[curr_layer + 2] + 1
        print(start_layer, mid_layer, end_layer)

        layers = []
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                if display:
                    print(layer)
                l = copy.deepcopy(layer)
                for p in l.parameters():
                    p.data = p.data.type(torch.DoubleTensor).cuda()
                layers.append(l)
        net = nn.Sequential(*layers)

        noise = init.type(torch.cuda.DoubleTensor).clone()
        target = Variable(feat.type(torch.cuda.DoubleTensor)).detach()


        noise_size = noise.size()

        # ================
        def go(x):
            x = x.view(noise_size)
            output = net(x)
            se = torch.sum((output - target) ** 2)
            return se

        def f(x):
            assert torch.is_tensor(x)
            x = Variable(x, requires_grad=True)
            loss = go(x)
            loss.backward()
            grad = x.grad.data.view(-1)
            return loss.data[0], grad

        # ================
        noise = noise.view(-1)
        for idx in range(1000):
            noise, stat = lbfgs(f, noise, maxIter=iters, gEps=1e-4, histSize=4, lr=lr, display=display)
            if stat in ["LBFGS REACH MAX ITER", "LBFGS BELOW GRADIENT EPS"]:
                print(stat)
                break


        noise = noise.type(torch.cuda.FloatTensor)
        noise = Variable(noise.view(noise_size), volatile=True)
        out = self.forward_subnet(input_tensor=noise, start_layer=start_layer, end_layer=mid_layer)

        return out.data




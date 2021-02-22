#!/usr/bin/python
import sys
import argparse

import torch
import scipy.ndimage as ndi
from torch import nn, optim, autograd
import paths
import helpers
from torch.autograd import Variable
import diymodel
import traceback
import numpy as np
import data
import os

default_degrade = "translation=0.03, rotation=1.0, scale=0.03, aniso=0.03"


parser = argparse.ArgumentParser("train a page segmenter")
parser.add_argument("-l", "--lr", default="0,0.03:3e5,0.01:1e6,0.003",
                    help="learning rate or learning rate sequence 'n,lr:n,lr:n,:r'")
parser.add_argument("-b", "--batchsize", type=int, default=1)
parser.add_argument("-o", "--output", default="temp", help="prefix for output")
parser.add_argument("-m", "--model", default=None, help="load model")
parser.add_argument("-d", "--input", default="uw3-framed-lines.tgz")

parser.add_argument("--maxtrain", type=int, default=10000000)
parser.add_argument("--degrade", default=default_degrade,
                    type=str, help="degradation parameters")
parser.add_argument("--erange", default=20, type=int,
                    help="line emphasis range")
parser.add_argument("--scale", default=1.0, type=float,
                    help="rescale prior to training")
parser.add_argument("--save_every", default=1000,
                    type=int, help="how often to save")
parser.add_argument("--loss_horizon", default=1000, type=int,
                    help="horizon over which to calculate the loss")
parser.add_argument("--dilate_target", default=0, type=int,
                    help="extra dilation for target")
parser.add_argument("--dilate_mask", default="(30,150)",
                    help="dilate of target to make mask")
parser.add_argument("--mask_background", default=0.0,
                    type=float, help="background weight for mask")
parser.add_argument("--ntrain", type=int, default=-
                    1, help="ntrain starting value")
parser.add_argument("--display", type=int, default=10,
                    help="how often to display samples and outputs")
parser.add_argument("--complexity", type=int, default=10,
                    help="base model complexity")

args = parser.parse_args()
ARGS = {k: v for k, v in args.__dict__.items()}


def pixels_to_batch(x):
    b, d, h, w = x.size()
    return x.permute(0, 2, 3, 1).contiguous().view(b*h*w, d)


class WeightedGrad(autograd.Function):

    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        return grad_output * weights


def weighted_grad(x, y):
    return WeightedGrad.apply(x, y)


# FIXME replace with version in dltrainers


class LearningRateSchedule(object):
    def __init__(self, schedule):
        if ":" in schedule:
            self.learning_rates = [
                [float(y) for y in x.split(",")] for x in schedule.split(":")]
            assert self.learning_rates[0][0] == 0
        else:
            lr0 = float(schedule)
            self.learning_rates = [[0, lr0]]

    def __call__(self, count):
        _, lr = self.learning_rates[0]
        for n, l in self.learning_rates:
            if count < n:
                break
            lr = l
        return lr


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

source = data.DocLayoutDataset("./image","./label", transform = transforms.Compose([data.ToTensor()]))
if args.model:
    model = torch.load(args.model)
    ntrain = 0
else:
    model = diymodel.LayoutModel(args.complexity)
    ntrain = 0
model.cuda()
if args.ntrain >= 0:
    ntrain = args.ntrain
print ("ntrain", ntrain)

print ("model:")
print (model)

start_count = 0

criterion = nn.MSELoss()
criterion.cuda()

losses = [1.0]


def zoom_like(image, shape):
    h, w = shape
    image = helpers.asnd(image)
    ih, iw = image.shape
    scale = np.diag([ih * 1.0/h, iw * 1.0/w])
    return ndi.affine_transform(image, scale, output_shape=(h, w), order=1)


def zoom_like_batch(batch, shape):
    batch = batch.numpy()
    b, h, w, d = batch.shape
    oh, ow = shape
    batch_result = []
    for i in range(b):
        result = []
        result.append(zoom_like(batch[i, 0, :, :], (oh, ow)))
        batch_result.append(result)
    result = np.array(batch_result)
    result = torch.FloatTensor(result)
    return result


def train_batch(model, image, target, mask=None, lr=1e-3):
    cuinput = image.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.0)
    optimizer.zero_grad()
    cuoutput = model(Variable(cuinput))
    b, d, h, w = cuoutput.size()
    if mask is not None:
        mask = zoom_like_batch(mask, (h, w))
        cumask = mask.cuda()
        coutput = weighted_grad(cuoutput, Variable(cumask))
    target = zoom_like_batch(target, (h, w))
    cutarget = Variable(target.cuda())
    loss = criterion(pixels_to_batch(cuoutput), pixels_to_batch(cutarget))
    loss.backward()
    optimizer.step()
    return loss.item(), helpers.asnd(cuoutput).transpose(0, 2, 3, 1)

def make_save_path(prefix, ntrain, error, extension="pt"):
    assert isinstance(ntrain, int)
    assert isinstance(error, float)
    assert ntrain < 1e12
    assert error >= -1e-6
    assert error <= 1+1e-6
    if extension is not None:
        extension = "." + extension
    error = clip(error, 0, 1)
    kilos = int(ntrain // 1000)
    micros = int(error * 1e6)
    return prefix + "-{:09d}-{:06d}".format(kilos, micros) + extension

losses = []
rates = LearningRateSchedule(args.lr)
nbatches = 0
from numpy import clip
best_loss = 0

for iter in range(args.maxtrain):
    print ("iter： " + str(iter))
    num = 0
    dataloader = DataLoader(source, batch_size=2, shuffle=True, num_workers=1)
    for i_batch, sample in enumerate(dataloader):
        fname = sample["name"]
        image = sample["image"]
        target = sample["label"]
        mask = sample.get("mask")
        lr = rates(ntrain)
        try:
            loss, output = train_batch(model, image, target, mask, lr)
        except Exception as e:
            utils.print_sample(sample)
            print (traceback.format_exc())
            continue
        
        losses.append(loss)
        print (nbatches, ntrain, i_batch,)
        print (loss, fname, np.amin(output), np.amax(output), "lr", lr)
        if ntrain > 0 and ntrain % args.save_every == 0:
            err = float(np.mean(losses[-args.save_every:]))
            fname = make_save_path(args.output, ntrain, err)
            if best_loss == 0 or loss < best_loss:
                best_loss = loss
                torch.save(model, fname)
                print ("saved", fname)
        nbatches += 1
        ntrain += len(image)
        sys.stdout.flush()
        sys.stderr.flush()

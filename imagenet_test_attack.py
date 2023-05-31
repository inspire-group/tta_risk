from ast import Pass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models 

from pathlib import Path

import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import logging
import os

from tqdm import tqdm
import numpy as np

from models import *
from conf import cfg, load_cfg_fom_args

from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_cifar10c, load_cifar10, load_cifar100c, load_cifar10
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
from utils.imagenetloader import CustomImageFolder

import tent
import copy
import bn 

from utils import get_imagenet_r_mask 


torch.manual_seed(0)

from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

torch.backends.cudnn.enabled=True


logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cfg_fom_args('"ImageNet evaluation.')
logger.info("test-time adaptation:")

imagenet_r_mask = get_imagenet_r_mask()


if not os.path.exists(cfg.LOG_DIR):
    os.makedirs(cfg.LOG_DIR)


if cfg.MODEL.ARCH == "Standard_R50":
    os.environ['TORCH_HOME'] = cfg.CKPT_DIR
    print(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    net = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

elif cfg.MODEL.ARCH == "Hendry_R50":
    os.environ['TORCH_HOME'] = cfg.CKPT_DIR
    net = load_model("Hendrycks2020Many", cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

elif cfg.MODEL.ARCH == "Augmix_R50":
    os.environ['TORCH_HOME'] = cfg.CKPT_DIR
    net = load_model("Hendrycks2020AugMix", cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

elif cfg.MODEL.ARCH == "Salman_R50":
    os.environ['TORCH_HOME'] = cfg.CKPT_DIR
    net = load_model("Salman2020Do_R50", cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, 'Linf').cuda()

elif cfg.MODEL.ARCH == "Engstrom_R50":
    os.environ['TORCH_HOME'] = cfg.CKPT_DIR
    net = load_model("Engstrom2019Robustness", cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, 'Linf').cuda()

else:
    pass 


def setup_optimizer(params, lr_test=None):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if lr_test is None:
        lr_test = cfg.OPTIM.LR

    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=lr_test,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=lr_test,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def meta_test_adaptive(model, test_loader, n_inner_iter=1, adaptive=True, update = False,  num_classes=1000):
    if cfg.MODEL.ADAPTATION == "EMABN":
        model = bn.adapt_ema(model)
    elif cfg.MODEL.ADAPTATION == "PARTBN":
        adapt_mean, adapt_var = False, True
        model = bn.adapt_parts(model,adapt_mean, adapt_var)
    elif cfg.MODEL.ADAPTATION == "BAYESBN":
        model = bn.adapt_bayesian(model, cfg.ATTACK.DFPIROR, cfg.ATTACK.Layer)
    elif cfg.MODEL.ADAPTATION == "RBN":
        model = bn.adapt_robustBN(model, cfg.ATTACK.DFPIROR, cfg.ATTACK.Layer)
    elif cfg.MODEL.ADAPTATION == "MBN":
        model = bn.adapt_MBN(model)
    else:
        model = tent.configure_model(model)

        
    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    if not adaptive:
        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)

    acc = 0.
    counter = 0 
    num_examples = 0

    iterator = tqdm(test_loader)
    for batch_data in iterator:
        if cfg.TEST.DATASET == "imagenetc" :
            x_curr, y_curr, _ = batch_data 
        elif cfg.TEST.DATASET == "imagenetr" or cfg.TEST.DATASET == "imagenet":
            x_curr, y_curr = batch_data 

        counter += 1

        num_examples += x_curr.shape[0]
        if counter % 50 == 0:
            print("batch id ", counter)

        if not adaptive:
            load_model_and_optimizer(model, inner_opt,
                                 model_state, optimizer_state)

        x_curr, y_curr = x_curr.cuda(), y_curr.cuda()
        y_curr = y_curr.type(torch.cuda.LongTensor)
        
        if update:
            for _ in range(n_inner_iter):
                T = cfg.OPTIM.TEMP
                eps = cfg.MODEL.EPS
                # with torch.no_grad():
                outputs = model(x_curr)

                if cfg.TEST.DATASET == "imagenetr":
                    outputs = outputs[:, imagenet_r_mask]

                outputs = outputs / T
                if cfg.OPTIM.ADAPT == "ent":
                    tta_loss = softmax_entropy(outputs)
                elif cfg.OPTIM.ADAPT == "rpl":
                    p = F.softmax(outputs, dim=1)
                    y_pl = outputs.max(1)[1]
                    Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
                    tta_loss = (1- (Yg**0.8))/0.8
                elif cfg.OPTIM.ADAPT == "conjugate":
                    softmax_prob = F.softmax(outputs, dim=1)
                    smax_inp = softmax_prob 

                    eye = torch.eye(num_classes).to(outputs.device)
                    eye = eye.reshape((1, num_classes, num_classes))
                    eye = eye.repeat(outputs.shape[0], 1, 1)
                    t2 = eps * torch.diag_embed(smax_inp)
                    smax_inp = torch.unsqueeze(smax_inp, 2)
                    t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
                    matrix = eye + t2 - t3
                    y_star = torch.linalg.solve(matrix, smax_inp)
                    y_star = torch.squeeze(y_star)

                    pseudo_prob = y_star
                    tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "softmax_pl":
                    softmax_prob = F.softmax(outputs, dim=1)

                    tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob *(1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "hard_pl":
                    yp = outputs.max(1)[1]
                    eps=8
                    y_star = 1 * F.one_hot(yp, num_classes=num_classes)
                    thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
                    tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
                else:
                    tta_loss = None

                tta_loss = tta_loss.mean()

                inner_opt.zero_grad()
                tta_loss.backward()

                inner_opt.step()
        
        with torch.no_grad():
            outputs_new = model(x_curr)

        if num_classes == 200:
            outputs_new = outputs_new[:, imagenet_r_mask]
        #print(outputs_new.max(1)[1], y_curr)

        acc += (outputs_new.max(1)[1] == y_curr).float().sum()
        print(acc/num_examples) 

    return acc.item() / num_examples



def get_imagenetc_loader(data_dir, corruption, severity, batch_size, shuffle=False):
    data_folder_path = Path(data_dir) / "ImageNet-C"/ corruption / str(severity)

    prepr = transforms.Compose([
        transforms.ToTensor()
    ])
    imagenet = CustomImageFolder(data_folder_path, prepr)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=1)
    return test_loader 

def get_imagenet_loader(data_dir, batch_size, shuffle=False):
    data_folder_path = Path(data_dir) / "val"

    prepr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    imagenet = datasets.ImageFolder(root=data_folder_path, transform=prepr)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=1)
    return test_loader 

def get_imagenetr_loader(data_dir, batch_size, shuffle=False):
    prepr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    imagenet_r = datasets.ImageFolder(root=data_dir, transform=prepr)

    test_loader = data.DataLoader(imagenet_r,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=1,
                                  pin_memory=True)
    return test_loader 


def test_attack_adaptive(model, test_loader, batch_size,  n_inner_iter=1, adaptive=True, use_test_bn=True, num_classes=1000,update =True):

    if cfg.MODEL.ADAPTATION == "EMABN":
        model = bn.adapt_ema(model)
    elif cfg.MODEL.ADAPTATION == "PARTBN":
        adapt_mean, adapt_var = False, True
        model = bn.adapt_parts(model,adapt_mean, adapt_var)
    elif cfg.MODEL.ADAPTATION == "BAYESBN":
        model = bn.adapt_bayesian(model, cfg.ATTACK.DFPIROR, cfg.ATTACK.Layer)
    elif cfg.MODEL.ADAPTATION == "RBN":
        model = bn.adapt_robustBN(model, cfg.ATTACK.DFPIROR, cfg.ATTACK.Layer)
    elif cfg.MODEL.ADAPTATION == "MBN":
        model = bn.adapt_MBN(model)
    else:
        if use_test_bn:
            model = tent.configure_model(model)
        else:
            model = tent.configure_model_eval(model)
        


    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    acc_target_be_all, acc_target_af_all, acc_clean_all, acc_adv_all,acc_source_be_all,acc_source_af_all = 0.,0.,0.,0.,0.,0.

    victim_model = copy.deepcopy(model).cuda()
    params_victim, _ = tent.collect_params(victim_model)
    inner_opt_victim = setup_optimizer(params_victim)

    if cfg.ATTACK.WHITE:
        sur_model = copy.deepcopy(model).cuda()
        params_sur, _ = tent.collect_params(sur_model)
        inner_opt_sur = setup_optimizer(params_sur)

    attack = ATTACK(cfg, source = cfg.ATTACK.SOURCE, target=cfg.ATTACK.TARGET , num_classes=num_classes)
    num_examples = 0
    counter = 0
    for batch_data in test_loader:
        if cfg.TEST.DATASET == "imagenetc":
            x_curr, y_curr, _ = batch_data 
        elif cfg.TEST.DATASET == "imagenetr":
            x_curr, y_curr = batch_data 
        print(y_curr)
        counter += 1
        num_examples += y_curr.shape[0]
        x_curr, y_curr = x_curr.cuda(), y_curr.cuda()

        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)
        load_model_and_optimizer(sur_model, inner_opt_sur, model_state, optimizer_state)
        load_model_and_optimizer(victim_model, inner_opt_victim, model_state, optimizer_state)

        if update:
            for _ in range(n_inner_iter):
                outputs = model(x_curr)
                if cfg.TEST.DATASET == "imagenetr":
                    outputs = outputs[:, imagenet_r_mask] 
                outputs = outputs / cfg.OPTIM.TEMP
                softmax_prob = F.softmax(outputs, dim=1)
                eps = cfg.MODEL.EPS
                if cfg.OPTIM.ADAPT == "ent":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                elif cfg.OPTIM.ADAPT == "conjugate":
                    smax_inp = softmax_prob 

                    eye = torch.eye(num_classes).to(outputs.device)
                    eye = eye.reshape((1, num_classes, num_classes))
                    eye = eye.repeat(outputs.shape[0], 1, 1)
                    t2 = eps * torch.diag_embed(smax_inp)
                    smax_inp = torch.unsqueeze(smax_inp, 2)
                    t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
                    matrix = eye + t2 - t3
                    y_star = torch.linalg.solve(matrix, smax_inp)
                    y_star = torch.squeeze(y_star)

                    pseudo_prob = y_star
                    tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "softmax_pl":
                    tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob * (1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "hard_pl":
                    yp = outputs.max(1)[1]
                    y_star = 1 * F.one_hot(yp, num_classes=num_classes)
                    thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
                    tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
                elif cfg.OPTIM.ADAPT == "rpl":
                    p = F.softmax(outputs, dim=1)
                    y_pl = outputs.max(1)[1]
                    Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
                    tta_loss = (1- (Yg**0.8))/0.8
                else:
                    pass 
                
                tta_loss = tta_loss.mean()
                inner_opt.zero_grad()
                tta_loss.backward()
                inner_opt.step()
        with torch.no_grad():
            outputs_clean = model(x_curr)
        if cfg.TEST.DATASET == "imagenetr":
            outputs_clean = outputs_clean[:, imagenet_r_mask]
        attack.update_target(outputs_clean, y_curr,counter)
        if cfg.ATTACK.METHOD == "PGD":
            if cfg.ATTACK.ADAPTIVE:
                pass
            else:
                x_adv = attack.generate_attacks( sur_model= sur_model, x= x_curr, y = y_curr,
                    randomize=False, epsilon=cfg.ATTACK.EPS, 
                    alpha=cfg.ATTACK.ALPHA, num_iter=cfg.ATTACK.STEPS)

        else:
            pass

        if update:
            for _ in range(n_inner_iter):
                outputs = victim_model(x_adv)
                if cfg.TEST.DATASET == "imagenetr":
                    outputs = outputs[:, imagenet_r_mask] 

                outputs = outputs / cfg.OPTIM.TEMP
                softmax_prob = F.softmax(outputs, dim=1)
                eps = cfg.MODEL.EPS

                if cfg.OPTIM.ADAPT == "ent":
                    tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                elif cfg.OPTIM.ADAPT == "conjugate":
                    smax_inp = softmax_prob 

                    eye = torch.eye(num_classes).to(outputs.device)
                    eye = eye.reshape((1, num_classes, num_classes))
                    eye = eye.repeat(outputs.shape[0], 1, 1)
                    t2 = eps * torch.diag_embed(smax_inp)
                    smax_inp = torch.unsqueeze(smax_inp, 2)
                    t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
                    matrix = eye + t2 - t3
                    y_star = torch.linalg.solve(matrix, smax_inp)
                    y_star = torch.squeeze(y_star)

                    pseudo_prob = y_star
                    tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "softmax_pl":
                    tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob * (1-softmax_prob)).sum(dim=1)
                elif cfg.OPTIM.ADAPT == "hard_pl":
                    yp = outputs.max(1)[1]
                    y_star = 1 * F.one_hot(yp, num_classes=num_classes)
                    thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
                    tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
                elif cfg.OPTIM.ADAPT == "rpl":
                    p = F.softmax(outputs, dim=1)
                    y_pl = outputs.max(1)[1]
                    Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
                    tta_loss = (1- (Yg**0.8))/0.8
                else:
                    pass 
                tta_loss = tta_loss.mean()
                inner_opt_victim.zero_grad()
                tta_loss.backward()
                inner_opt_victim.step()
        with torch.no_grad():
            outputs_adv = victim_model(x_adv)
        if cfg.TEST.DATASET == "imagenetr":
            outputs_adv = outputs_adv[:, imagenet_r_mask]

        acc_target_be, acc_target_af, acc_clean, acc_adv, acc_source_be, acc_source_af =attack.compute_acc(outputs_clean,outputs_adv,y_curr)
        acc_target_be_all += acc_target_be
        acc_target_af_all += acc_target_af
        acc_clean_all += acc_clean
        acc_adv_all += acc_adv
        acc_source_be_all += acc_source_be
        acc_source_af_all += acc_source_af
        if cfg.DEBUG:
            print("acc_target_be: ", acc_target_be.item(), "acc_target_af: ", acc_target_af.item(), "acc_clean: ", 
            acc_clean.item(), "acc_adv: ", acc_adv.item(), "acc_source_be: ", acc_source_be.item(), "acc_source_af: ", 
            acc_source_af.item())
    print(counter,num_examples)
    logger.info(f"target accuracy: % [{corruption_type}{severity}]: {(acc_target_be_all.item()/counter):.2%}")
    logger.info(f"target accuracy attack: % [{corruption_type}{severity}]: {(acc_target_af_all.item()/counter):.2%}")
    logger.info(f"clean accuracy: % [{corruption_type}{severity}]: {(acc_clean_all.item()/num_examples):.2%}")
    logger.info(f"adv accuracy: % [{corruption_type}{severity}]: {(acc_adv_all.item()/num_examples):.2%}")
    logger.info(f"source accuracy: % [{corruption_type}{severity}]: {(acc_source_be_all.item()/counter/cfg.ATTACK.SOURCE):.2%}")
    logger.info(f"source adv accuracy: % [{corruption_type}{severity}]: {(acc_source_af_all.item()/counter/cfg.ATTACK.SOURCE):.2%}")



class ATTACK():
    def __init__(self, cfg, source, target, num_classes):
        self.cfg = cfg
        self.source = source
        self.target = target
        self.num_classes = num_classes
    
    def update_target(self, outputs_clean, y, counter):
        target = self.target

        if self.cfg.ATTACK.TARGETED:
            self.target = 0 
            self.target_label = (y[self.target]+1)%self.num_classes 
        else:
            acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
            while acc_target_be.item() == 0.:
                target += 1
                acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
                if target > self.cfg.TEST.BATCH_SIZE - self.source - 1:
                    target = 0
            self.target = target

        self.counter = counter

        # if self.cfg.ATTACK.TARGETED:
        #     self.target = 0 
        #     self.target_label = (y[self.target]+1)%self.num_classes 
            # print(y[self.target], "target label: ", self.target_label)

    def generate_attacks(self, sur_model, x, y, randomize=False, 
        epsilon=16/255, alpha=2/255, num_iter=10):
        source = self.source
        target = self.target
        fixed = torch.zeros_like(x.clone()[:-source], requires_grad=False)
        adv = (torch.zeros_like(x.clone()[-source:]) - x[-source:] + 127.5/255 ).requires_grad_(True)
        adv_pad = torch.cat((fixed, adv), 0)

        if self.cfg.ATTACK.TARGETED:
            for t in tqdm(range(num_iter),disable=True):
                x_adv = x + adv_pad
                out = sur_model(x_adv)
                if self.cfg.TEST.DATASET == "imagenetr":
                    out = out[:, imagenet_r_mask]
                loss = nn.CrossEntropyLoss(reduction='none')(out[target], self.target_label)
                loss.backward()
                if cfg.DEBUG:
                    print("loss", loss.item(), "out", out[target].argmax().item(), "y", y[target].item(), 'target_label', self.target_label)     
                adv.data = (adv - alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
                adv.data = (adv.data +x[-source:]).clamp(0,1)-(x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0) 
                adv.grad.zero_()
        else:
            for t in tqdm(range(num_iter),disable=True):
                x_adv = x + adv_pad
                out = sur_model(x_adv)
                if self.cfg.TEST.DATASET == "imagenetr":
                    out = out[:, imagenet_r_mask]
                loss = nn.CrossEntropyLoss(reduction='none')(out[target], y[target])
                loss.backward()
                print('Learning Progress :%2.2f %% , loss1 : %f ' %((t+1)/num_iter*100, loss.item()), end='\r')

                if cfg.DEBUG:
                    print("loss", loss.item(), "out", out[target].argmax().item(), "y", y[target].item())
                
                adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
                adv.data = (adv.data +x[-source:]).clamp(0,1)-(x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0) 
                adv.grad.zero_()

        x_adv = x + adv_pad
        return x_adv

    def get_tta_loss(self,out):
        cfg = self.cfg
        num_classes = self.num_classes
        outputs = out / cfg.OPTIM.TEMP
        softmax_prob = F.softmax(outputs, dim=1)
        eps = cfg.MODEL.EPS
        if cfg.OPTIM.ADAPT == "ent":
            tta_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        elif cfg.OPTIM.ADAPT == "conjugate":
            smax_inp = softmax_prob
            eye = torch.eye(num_classes).to(outputs.device)
            eye = eye.reshape((1, num_classes, num_classes))
            eye = eye.repeat(outputs.shape[0], 1, 1)
            t2 = eps * torch.diag_embed(smax_inp)
            smax_inp = torch.unsqueeze(smax_inp, 2)
            t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
            matrix = eye + t2 - t3
            y_star = torch.linalg.solve(matrix, smax_inp)
            y_star = torch.squeeze(y_star)
            pseudo_prob = y_star
            tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
        elif cfg.OPTIM.ADAPT == "softmax_pl":
            tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob * (1-softmax_prob)).sum(dim=1)
        elif cfg.OPTIM.ADAPT == "hard_pl":
            yp = outputs.max(1)[1]
            y_star = 1 * F.one_hot(yp, num_classes=num_classes)
            thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
            tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
        elif cfg.OPTIM.ADAPT == "rpl":
            p = F.softmax(outputs, dim=1)
            y_pl = outputs.max(1)[1]
            Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
            tta_loss = (1- (Yg**0.8))/0.8
        else:
            pass 
        return tta_loss

    def compute_acc(self,outputs_clean, outputs_adv, y):
        target = self.target
        source = self.source
        acc_target_be = (outputs_clean[target].argmax() == y[target]).float()
        acc_source_be = (outputs_clean.max(1)[1][-source:] == y[-source:]).float().sum()
        acc_clean = (outputs_clean.max(1)[1] == y).float().sum()
        acc_adv = (outputs_adv.max(1)[1] == y).float().sum()
        acc_target_af = (outputs_adv[target].argmax() == y[target]).float()
        acc_source_af = (outputs_adv.max(1)[1][-source:] == y[-source:]).float().sum()
        if self.cfg.ATTACK.TARGETED:
            acc_target_af = (outputs_adv[target].argmax() == self.target_label).float()
        print("before",  outputs_clean[target].argmax().item(),
              "after" ,  outputs_adv[target].argmax().item(), 
              "correct", y[target].item(),
              "loss",    nn.CrossEntropyLoss(reduction='none')(outputs_adv, y)[target].item(),
              "loss_clean",    nn.CrossEntropyLoss(reduction='mean')(outputs_clean, y).item())
        return acc_target_be, acc_target_af, acc_clean, acc_adv, acc_source_be, acc_source_af


err_array = np.zeros((len(cfg.CORRUPTION.SEVERITY)+1, len(cfg.CORRUPTION.TYPE)+1))
save_path = os.path.join(cfg.LOG_DIR, "adapt_%s_opt_%s_lr_%.1e_T_%.1f.txt"%(cfg.OPTIM.ADAPT, cfg.OPTIM.METHOD, cfg.OPTIM.LR, cfg.OPTIM.TEMP))
np.savetxt(save_path, err_array, fmt="%.4f")


for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        if cfg.TEST.DATASET == "imagenetc":
            test_loader = get_imagenetc_loader("../data", corruption_type, severity, cfg.TEST.BATCH_SIZE, False)
            num_classes = 1000
        elif cfg.TEST.DATASET == "imagenetr":
            test_loader = get_imagenetr_loader("../data/ImageNet-R/imagenet-r", cfg.TEST.BATCH_SIZE, False)
            num_classes = 200
        elif cfg.TEST.DATASET == "imagenet":
            test_loader = get_imagenet_loader("../data/ImageNet/", cfg.TEST.BATCH_SIZE, False)
            num_classes = 1000

        print("Meta test begin!")
        net_test = copy.deepcopy(net)

        if cfg.ATTACK.METHOD == None:
            acc = meta_test_adaptive(net_test, test_loader, cfg.OPTIM.STEPS, adaptive=False, update=cfg.OPTIM.UPDATE, num_classes=num_classes)
            logger.info(f"acc % [{corruption_type}{severity}]: {acc:.2%}")
        else:
            test_attack_adaptive(net_test, test_loader, cfg.TEST.BATCH_SIZE, cfg.OPTIM.STEPS, adaptive=cfg.OPTIM.ADAPTIVE, use_test_bn=cfg.OPTIM.TBN, num_classes=num_classes, update=cfg.OPTIM.UPDATE)
        
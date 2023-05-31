import os 
from glob import escape
from re import U
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import logging 
import numpy as np
from models import * 
from conf import cfg, load_cfg_fom_args
from robustbench.data import load_cifar10c, load_cifar100c
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
import tent
import copy
import bn
import time

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
np.random.seed(0)


from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

torch.backends.cudnn.enabled=True

from pdb import set_trace as st 

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import torchvision
import matplotlib.pyplot as plt
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imsave("images/test"+title+".png",inp)

load_cfg_fom_args('"CIFAR-10-C evaluation.')
logger.info("test-time adaptation: TENT")

if not os.path.exists(cfg.LOG_DIR):
    os.makedirs(cfg.LOG_DIR)

if cfg.CORRUPTION.DATASET == "cifar10":
    if cfg.MODEL.ARCH == 'resnet26':
        ckpt_path = "saved_models/pretrained/cifar10_ce.pth" # edit this path to the checkpoint of the model you want to evaluate
        net = Normalized_ResNet(depth=26)
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint["net"])
        net = net.module.to(device)

    elif cfg.MODEL.ARCH == 'Standardwrn28': 
        name = 'Standard'
        net = load_model(model_name=name,dataset='cifar10',threat_model='Linf').to(device)

    elif cfg.MODEL.ARCH == 'VGG': 
        ckpt_path ="saved_models/pretrained/cifar10_ce_vgg.pth" # edit this path to the checkpoint of the model you want to evaluate
        net = Normalized_VGG_CIFAR10().to(device)
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint["net"])
        net = net.to(device)

    elif cfg.MODEL.ARCH == 'Gowalwrn28':
        name = 'Gowal2021Improving_28_10_ddpm_100m'
        net = load_model(model_name=name,dataset='cifar10',threat_model='Linf').to(device)

    elif cfg.MODEL.ARCH == 'Wuwrn28': 
        name = 'Wu2020Adversarial_extra'
        net = load_model(model_name=name,dataset='cifar10',threat_model='Linf').to(device)


    elif cfg.MODEL.ARCH == 'Sehwagrn18':
        name = 'Sehwag2021Proxy_R18'
        net = load_model(model_name=name,dataset='cifar10',threat_model='Linf').to(device)


    else:
        raise NotImplementedError

    cudnn.benchmark = True

elif cfg.CORRUPTION.DATASET == "cifar100":
    if cfg.MODEL.ARCH == 'resnet26':
        ckpt_path = "saved_models/pretrained/cifar100_ce.pth"
        net = Normalized_ResNet_CIFAR100()
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint["net"])
        net = net.module.to(device)

    elif cfg.MODEL.ARCH == 'Standardwrn28':
        ckpt_path ="saved_models/pretrained/cifar100_wrn_ce.pth"
        net = Normalized_WideResNet_CIFAR100().to(device)
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint["net"])
        net = net.to(device)

    elif cfg.MODEL.ARCH == 'VGG': 
        ckpt_path ="saved_models/pretrained/cifar100_vgg_ce.pth"
        net = Normalized_VGG_CIFAR100().to(device)
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint["net"])
        net = net.to(device)

    elif cfg.MODEL.ARCH == 'Pangwrn28':
        name = 'Pang2022Robustness_WRN28_10'
        net = load_model(model_name=name,dataset='cifar100',threat_model='Linf').to(device)

    elif cfg.MODEL.ARCH == 'Rebuffwrn28':
        name = 'Rebuffi2021Fixing_28_10_cutmix_ddpm'
        net = load_model(model_name=name,dataset='cifar100',threat_model='Linf').to(device)
        
    else:
        raise NotImplementedError

    cudnn.benchmark = True

def test_clean(model, x_test, y_test, batch_size):
    acc = 0.
    model.eval()

    n_batches = math.ceil(x_test.shape[0] / batch_size)
    for counter in range(n_batches):
        x_curr = x_test[counter * batch_size:(counter + 1) *
                   batch_size].to(device)
        y_curr = y_test[counter * batch_size:(counter + 1) *
                   batch_size].to(device)

        outputs = model(x_curr)
        acc += (outputs.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
    return model

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

def meta_test_adaptive(model, x_test, y_test, batch_size,  n_inner_iter=1, adaptive=True, use_test_bn=True, num_classes=10,update =True):
    if use_test_bn:
        model = tent.configure_model(model)
    else:
        model = tent.configure_model_eval(model)

    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    acc = 0.
    n_batches = math.ceil(x_test.shape[0] / batch_size)

    for counter in range(n_batches):
        x_curr = x_test[counter * batch_size:(counter + 1) * batch_size].to(device)
        y_curr = y_test[counter * batch_size:(counter + 1) * batch_size].to(device)
       
        if update:
            for _ in range(n_inner_iter):
                outputs = model(x_curr)
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


        outputs_new = model(x_curr)
        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]

def test_attack_adaptive(model, x_test, y_test, batch_size,  n_inner_iter=1, adaptive=True, use_test_bn=True, num_classes=10,update =True):

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

    n_batches = math.ceil(x_test.shape[0] / batch_size)
    acc_target_be_all, acc_target_af_all, acc_clean_all, acc_adv_all,acc_source_be_all,acc_source_af_all = 0.,0.,0.,0.,0.,0.

    victim_model = copy.deepcopy(model).cuda()
    params_victim, _ = tent.collect_params(victim_model)
    inner_opt_victim = setup_optimizer(params_victim)

    if cfg.ATTACK.WHITE:
        sur_model = copy.deepcopy(model).cuda()
        params_sur, _ = tent.collect_params(sur_model)
        inner_opt_sur = setup_optimizer(params_sur)

    attack = ATTACK(cfg, source = cfg.ATTACK.SOURCE, target=cfg.ATTACK.TARGET , num_classes=num_classes)

    for counter in range(n_batches):

        x_curr = x_test[counter * batch_size:(counter + 1) * batch_size].to(device)
        y_curr = y_test[counter * batch_size:(counter + 1) * batch_size].to(device)

        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)
        load_model_and_optimizer(sur_model, inner_opt_sur, model_state, optimizer_state)
        load_model_and_optimizer(victim_model, inner_opt_victim, model_state, optimizer_state)

        if update:
            for _ in range(n_inner_iter):
                outputs = model(x_curr)
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

    logger.info(f"target accuracy: % [{corruption_type}{severity}]: {(acc_target_be_all.item()/n_batches):.2%}")
    logger.info(f"target accuracy attack: % [{corruption_type}{severity}]: {(acc_target_af_all.item()/n_batches):.2%}")
    logger.info(f"clean accuracy: % [{corruption_type}{severity}]: {(acc_clean_all.item()/x_test.shape[0]):.2%}")
    logger.info(f"adv accuracy: % [{corruption_type}{severity}]: {(acc_adv_all.item()/x_test.shape[0]):.2%}")
    logger.info(f"source accuracy: % [{corruption_type}{severity}]: {(acc_source_be_all.item()/n_batches/10):.2%}")
    logger.info(f"source adv accuracy: % [{corruption_type}{severity}]: {(acc_source_af_all.item()/n_batches/10):.2%}")



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
                loss = nn.CrossEntropyLoss(reduction='none')(out[target], y[target])
                loss.backward()
                # if loss.item() > 1:
                #     break
                print('Learning Progress :%2.2f %% , loss1 : %f ' %((t+1)/num_iter*100, loss.item()), end='\r')

                if cfg.DEBUG:
                    print("loss", loss.item(), "out", out[target].argmax().item(), "y", y[target].item())
                
                adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
                adv.data = (adv.data +x[-source:]).clamp(0,1)-(x[-source:])
                adv_pad.data = torch.cat((fixed, adv), 0) 
                adv.grad.zero_()

        print(loss.item())
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





for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):

        if cfg.CORRUPTION.DATASET == "cifar10":
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,severity, cfg.DATA_DIR, False, [corruption_type])
            num_classes=10
            
        elif cfg.CORRUPTION.DATASET == "cifar100":
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,severity, cfg.DATA_DIR, False, [corruption_type])
            num_classes=100
        else:
            print("ERROR: no valid datatset provided, must be cifar10 and cifar100")

        x_test, y_test = x_test, y_test
        y_test = y_test.type(torch.LongTensor)



        print("Meta test begin!")
        net_test = copy.deepcopy(net)
        if cfg.ATTACK.METHOD == None:
            acc = meta_test_adaptive(net_test, x_test, y_test, cfg.TEST.BATCH_SIZE, cfg.OPTIM.STEPS, adaptive=cfg.OPTIM.ADAPTIVE, use_test_bn=cfg.OPTIM.TBN, num_classes=num_classes, update=cfg.OPTIM.UPDATE)
            logger.info(f"accuracy % [{corruption_type}{severity}]: {acc:.2%}")
        else:
             test_attack_adaptive(net_test, x_test, y_test, cfg.TEST.BATCH_SIZE, cfg.OPTIM.STEPS, adaptive=cfg.OPTIM.ADAPTIVE, use_test_bn=cfg.OPTIM.TBN, num_classes=num_classes, update=cfg.OPTIM.UPDATE)
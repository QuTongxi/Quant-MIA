import torch
from torch import nn
from quantizer import *
from quant_utils import *


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: Log3Quantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.sym = uaq.sym
        # self.delta = nn.Parameter(uaq.delta,requires_grad=False)
        # self.zero_point = nn.Parameter(uaq.zero_point,requires_grad=False)    
        # self.n_levels = nn.Parameter(torch.tensor(uaq.n_levels),requires_grad=False)
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
        
def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        if isinstance(m, QuantModule):
            m.set_quant_state(weight_quant, act_quant)
            
                    
def layer_reconstruction(model, layer: QuantModule, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = True, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.2, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """

    set_quant_state(model, False, False)
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                   weight_tensor=layer.org_weight.data)
        layer.weight_quantizer.soft_targets = True

        # Set up optimizer
        opt_params = [layer.weight_quantizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        opt_params = [layer.act_quantizer.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
    else:
        cached_grads = None

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 4999 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
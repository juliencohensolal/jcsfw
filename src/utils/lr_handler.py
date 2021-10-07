import utils.c_logging as c_logging

from tensorflow.keras import optimizers

LOG = c_logging.getLogger(__name__)


def get_learning_rate(conf):
    lr = None
    if conf.lr == "CosineDecay": # Slow decay with start and finish LR (as % of starting LR)
        lr = optimizers.schedules.CosineDecay(
            initial_learning_rate=conf.lr_start, 
            decay_steps=conf.lr_decay_epochs, 
            alpha=conf.lr_alpha)
    elif conf.lr == "CosineDecayRestarts": # Slow decay with restarts
        lr = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=conf.lr_start, 
            first_decay_steps=conf.lr_decay_epochs, 
            alpha=conf.lr_alpha, 
            t_mul=conf.lr_restart_epoch_multiplier, 
            m_mul=conf.lr_restart_multiplier)
    elif conf.lr == "PiecewiseConstantDecay": # custom schedule with boundaries and associated values
        lr = optimizers.schedules.PiecewiseConstantDecay(
            boundaries=conf.lr_custom_boundaries,
            values=conf.lr_custom_values)
    elif conf.lr == "RampUp": # With transfer learning, starting with high LR would break the pre-trained weights
        lr = build_ramp_up(
            conf.lr_start, conf.lr_rampup_max, conf.lr_rampup_min, conf.lr_rampup_epochs, 
            conf.lr_rampup_sustain_epochs, conf.lr_rampup_decay_factor)
    else:
        LOG.error("Unknown learning rate, quitting")
    return lr


def build_ramp_up(lr_start, lr_max, lr_min, lr_rampup_epochs, lr_sustain_epochs, lr_exp_decay):

    def ramp_up(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs- lr_sustain_epochs) + lr_min
        return lr
    
    return ramp_up

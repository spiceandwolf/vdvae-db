
HPARAMS_REGISTRY = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


power = Hyperparams()
power.width = 7
power.min_lr = 1e-6
power.zdim = 14
power.wd = 0.
# power.dec_blocks = "1x1,4m1,4x1,7m4,7x1" # x corresponds to residual block, m corresponds to unpool layer
# power.enc_blocks = "7x1,7d2,4x1,4d4,1x1" # x corresponds to residual block, d corresponds to pool layer
power.warmup_iters = 32
power.dataset = 'power'
power.n_batch = 2048
power.ema_rate = 0.9999
power.data_root = '/home/user/oblab/dataset/'
power.num_epochs = 20
power.desc = 'power_test'
power.skip_threshold = -1.
power.grad_clip = 30000
power.decay_iters = 36000
power.decay_start = 32
power.noise_value = '0.0005, 0.0005, 0.005, 0.05, 0.5, 0.5, 0.5'
power.encoding_modes = 'binary,binary,binary,binary,binary,binary,binary'
HPARAMS_REGISTRY['power'] = power

def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./')

    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--restore_ema_path', type=str, default=None)
    parser.add_argument('--restore_log_path', type=str, default=None)
    parser.add_argument('--restore_optimizer_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--ema_rate', type=float, default=0.999)

    parser.add_argument('--enc_blocks', type=str, default=None)
    parser.add_argument('--dec_blocks', type=str, default=None)
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--custom_width_str', type=str, default='')
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)

    parser.add_argument('--no_bias_above', type=int, default=64)
    parser.add_argument('--scale_encblock', action="store_true")

    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--warmup_iters', type=float, default=0)

    parser.add_argument('--num_mixtures', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)

    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--iters_per_ckpt', type=int, default=25000)
    parser.add_argument('--iters_per_print', type=int, default=1000)
    parser.add_argument('--iters_per_save', type=int, default=10000)
    parser.add_argument('--iters_per_images', type=int, default=10000)
    parser.add_argument('--epochs_per_eval', type=int, default=20)
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=8)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=3)
    
    parser.add_argument('--test_name', type=str, default=None)
    parser.add_argument('--noise_value', type=str, default=None)
    parser.add_argument('--noise_type', type=str, default='None')
    parser.add_argument('--normalize', type=str, default='minmax')
    parser.add_argument('--gradient_smoothing_beta', type=float, default=0.6931472)
    parser.add_argument('--last_epoch', type=int, default=-1)
    parser.add_argument('--decay_iters', type=int, default=0)
    parser.add_argument('--decay_start', type=int, default=0)
    parser.add_argument('--out_net_mode', type=str, default='')
    parser.add_argument('--mse_mode', type=str, default='')
    parser.add_argument('--std_mode', type=str, default='')
    parser.add_argument('--remarks', type=str, default='')
    parser.add_argument('--train_ray_tune', action="store_true")
    parser.add_argument('--tuning_recover', action="store_true")
    parser.add_argument('--discrete', action="store_true")
    parser.add_argument('--encoding_modes', type=str, default='')
    parser.add_argument('--vae_type', type=str, default='')
    
    return parser

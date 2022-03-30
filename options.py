import os
import argparse
import datetime

class BaseOptions:
    def __init__(self):
        self.parser = None

    def initialize(self, parser):

        parser.add_argument("--dataset", type=int, default=0, help="dataset: ntire dataset")
        parser.add_argument("--limit_training_dataset", type=float, default=1, help="limits the number of training samples")
        parser.add_argument("--limit_validation_dataset", type=float, default=1, help="limits the number of validation samples")
        parser.add_argument("--crop_train_data", type=int, default=None, help="Crops train data")
        parser.add_argument("--crop_test_data", type=int, default=1048, help="Crops test data")
        parser.add_argument('--not_augm', action='store_true', help="Disables data augmentation")
        parser.add_argument("--save_samples", type=int, default=15, help="Number of samples to save")
        parser.add_argument("--log_samples", type=int, default=5, help="Number of samples to save")
        parser.add_argument("--save_samples_every", type=int, default=5, help="Save samples every x epochs")
        parser.add_argument("--save_every", type=int, default=5, help="Save model every x epochs")
        parser.add_argument("--dataset_dir", type=str, default="/mnt/hdd/shared/datasets/ntire-hdr-2022-clean-256/", help='dataset directory')
        parser.add_argument('--init_kaiming', action='store_true', default=False, help='Initializes weights with kaiming')
        
        
        parser.add_argument("--name", type=str, help="Experiment name", default=None)
        parser.add_argument("--continue_from", type=str, help="Experiment name to continue from", default=None)

        parser.add_argument('--logdir', type=str, default='./checkpoints', help='target log directory')
        parser.add_argument('--workers', type=int, default=12,  help='number of workers to fetch data (default: 12)')

        # Training
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument("--gpu", type=int, help="GPU id, None: all", default=None)
        parser.add_argument('--seed', type=int, default=77, help='random seed (default: 77)')

        parser.add_argument("--loss", type=str, help="Loss to use", choices=["base", "mu"], default="base")
        parser.add_argument('--lambda_tanh', type=float, default=1, help="Lambda tanh loss")
        parser.add_argument('--lambda_pixel', type=float, default=1, help="Lambda pixel loss")
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument('--lr_decay_after', type=int, default=500, help='Learning rate decay after every N epochs(default: 500)')
        parser.add_argument('--start_epoch', type=int, default=0, help='start epoch of training (default: 1)')
        parser.add_argument('--continue_training', action='store_true', default=False, help='Continues the training from last state')
        parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
        parser.add_argument('--batch_size', '-bs', type=int, default=4, help='training batch size (default: 4)')

        # Model stuff
        parser.add_argument("--model", type=str, help="Model to use", choices=["drhdr"], default="drhdr")
        parser.add_argument('--base_channels', type=int, default=42)
        parser.add_argument('--input_channels', type=int, default=6)
        parser.add_argument('--dense_layers', type=int, default=6)
        parser.add_argument('--growth_rate', type=int, default=21)
        parser.add_argument('--last_residual', action='store_true', default=False, help='Uses residual instead of dense as the main skip connection')
        parser.add_argument('--deformable_groups', type=int, default=14)
        parser.add_argument('--eks', type=int, default=3, help="Encoder kernel size")
        parser.add_argument('--aks', type=int, default=3, help="Attention kernel size")
        parser.add_argument('--dks', type=int, default=3, help="Decoder kernel size")
        parser.add_argument('--activation', type=str, default="leaky", help="Activation function")


        # Inference
        parser.add_argument("--inference_test_dataset_dir", type=str, default="/mnt/hdd/shared/datasets/ntire-hdr-2022/ntire-test-input", help='Inference final test dataset directory')
        
        return parser
    
    def init(self):
        if self.parser is None:
            parser = argparse.ArgumentParser(description='HDR2022', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            return parser
        return self.parser

    def parse(self, args=None):
        self.parser = self.init()
            
        # get the basic options
        opt, unknown = self.parser.parse_known_args(args=args)
        opt = self.parser.parse_args(args=args)
        self.run_customs(opt)
        return opt
    
    def notebook(self, args=""):
        return self.parse(args)


    def prepare_logging_folder(self, args):
        # the default dir
        args.basedir = args.logdir
        timestamp = "log_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.logdir = os.path.join(args.logdir, args.name or timestamp)

        if args.continue_from is not None:
            args.continue_from = os.path.join(args.basedir, args.continue_from)

    def run_customs(self, args):
        self.prepare_logging_folder(args)


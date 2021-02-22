class base_options():

    def __init__(self, input_nc=4, output_nc=4):
        self.project_name = 'ecoli_100_1'
        self.data_path = 'others/'
        self.data_name = 'ecoli_100_1.csv'
        self.results_path = 'results/'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.split_ratio = 0.7
        self.ngf = 64
        self.ndf = 64
        self.gpu_ids = '0'
        self.netG = 'unet_128'
        self.netD = 'basic'
        self.gan_mode = 'lsgan'
        self.checkpoints_dir = 'checkpoints/'
        self.name = 'ecoli-1'
        self.verbose = True


class train_opt(base_options):

    def __init__(self, input_nc=4, output_nc=4):
        base_options.__init__(self, input_nc, output_nc)
        self.lr_policy = 'linear'
        self.gan_mode = 'lsgan'
        self.isTrain = True
        self.preprocess=''
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lr_decay_iters = 40
        self.lmd_l1 = 50
        self.continue_train = False
        self.log_dir = 'results/log_info.txt'
        self.n_epochs_decay = 50
        self.epoch_count = 1
        self.n_epochs = 50


class test_opt(base_options):
    def __init__(self, input_nc=4, output_nc=4):
        base_options.__init__(self, input_nc, output_nc)
        self.isTrain = False
        self.continue_train = False
        self.preprocess=''


class input_opt(base_options):

    def __init__(self, input_nc=4, output_nc=4):
        base_options.__init__(self, input_nc, output_nc)
        self.isTrain = False
        self.continue_train = False
        self.preprocess=''
        self.project_name = 'tet-100'
        self.data_path = 'others/'
        self.data_name = 'tet.csv'
        self.split_ratio = 0
        self.load_iter = 100

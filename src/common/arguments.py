import common.config as cfg


class Arguments():
    def __init__(
            self,
            args,
    ):
        # data config
        self.dataset = args.dataset
        self.clf = args.clf
        self.optim = args.optim
        self.paradigm = args.paradigm
        self.ncomponent = args.ncomponent
        self.rp_eps = args.rp_eps
        self.pca_var = args.pca_var
        self.sdir_full = args.sdir_full
        self.kgrads = args.kgrads
        self.topk = args.topk
        self.dga_bs = args.dga_bs
        self.num_dga = args.num_dga
        self.residual = args.residual
        self.error_tol = args.error_tol

        self.num_train = cfg.num_trains[self.dataset]*args.repeat
        self.num_test = cfg.num_tests[self.dataset]
        self.input_size = cfg.input_sizes[self.dataset]
        self.output_size = cfg.output_sizes[self.dataset]
        self.num_channels = cfg.num_channels[self.dataset]
        self.stratify = args.stratify
        self.num_workers = args.num_workers
        self.uniform_data = args.uniform_data
        self.shuffle_data = args.shuffle_data
        self.non_iid = args.non_iid
        self.repeat = args.repeat

        # training config
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        if not self.test_batch_size:
            self.test_batch_size = self.num_test
        self.noise = args.noise
        self.epochs = args.epochs
        self.lr = args.lr
        self.nesterov = args.nesterov
        self.decay = args.decay
        self.no_cuda = args.no_cuda
        self.device_id = args.device_id
        self.seed = args.seed

        # logging config
        self.log_intv = args.log_intv
        self.save_model = args.save_model
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        # dry run
        self.dry_run = args.dry_run
        if self.dry_run:
            self.save_model = False
            self.log_interval = 1
            self.epochs = 3
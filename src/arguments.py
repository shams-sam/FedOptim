import config as cfg


class Arguments():
    def __init__(
            self,
            args,
    ):
        # data config
        self.dataset = args.dataset
        self.clf = args.clf
        self.paradigm = args.paradigm
        self.num_comp = args.num_comp
        self.num_train = cfg.num_trains[self.dataset]*args.repeat
        self.num_test = cfg.num_tests[self.dataset]
        self.input_size = cfg.input_sizes[self.dataset]
        self.output_size = cfg.output_sizes[self.dataset]
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
        self.conj_dev = args.conj_dev
        self.kgrads = args.kgrads
        self.update_kgrads = args.update_kgrads
        self.no_cuda = args.no_cuda
        self.device_id = args.device_id
        self.seed = args.seed

        # logging config
        self.log_interval = args.log_interval
        self.save_model = args.save_model
        self.early_stopping = args.early_stopping
        # dry run
        self.dry_run = args.dry_run
        if self.dry_run:
            self.save_model = False
            self.log_interval = 1
            self.epochs = 2

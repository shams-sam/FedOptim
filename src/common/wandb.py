import wandb

# setup wandb/settings to give project, entity and api_key


def wandb_log(args):
    config = wandb.config
    for _ in args.__dict__:
        config.get(_) = args.get(_)

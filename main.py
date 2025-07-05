"""
main.py
-------
Entry point for running training or evaluation experiments with Beta-VAE models.
Handles argument parsing, configuration, and launching the appropriate workflow.
"""

import yaml
import logging
import argparse
import numpy as np
from torch import optim
import random
import sys
import time
import gc

from models.betaVAEHiggins import *
from models.betaVAEConv import *
from utils.datasets import *
from utils.helpers import *
from training import Trainer, save_model
from evaluate import Evaluator
from utils.visualize import Visualizer, GifTraversalsTraining
from utils.viz_new_plots import wandb_auth
from utils.viz_helpers import get_samples
from utils.beta_schedules import linear_beta_schedule

import wandb

RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
MODELS = ['BetaVAEHiggins', 'BetaVAEConv']

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-name', '--name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default="INFO", choices=LOG_LEVELS)
    
    general.add_argument('-s', '--seed', type=int, default=None,
                         help='Random seed. Can be `None` for stochastic behavior.')
    general.add_argument('--metrics_freq', type=int, default=-2,
                        help='The number of columns to visualize (if applicable).')
    general.add_argument('-max_traversal', '--max_traversal', type=float, default=0.475,
                         help='Random seed. Can be `None` for stochastic behavior.')
    general.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    general.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    general.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    general.add_argument("--plots", type=str, nargs='+', choices=PLOT_TYPES, default="all",
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. " \
                        "`reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses " \
                        "the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for " \
                        "original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are " \
                        "examples, each gif shows posterior traversals. `all` runs every plot.")
    general.add_argument('--n-rows', type=int, default=6,
                        help='The number of rows to visualize (if applicable).')
    general.add_argument('--n-cols', type=int, default=7,
                        help='The number of columns to visualize (if applicable).')
    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default='dsprites', choices=DATASETS)
    training.add_argument('-e', '--epochs', type=int,
                          default=10,
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=256,
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=0.01,
                          help='Learning rate.')

    training.add_argument('--dry_run', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,
                        help='Whether to use WANDB in offline mode.')
    training.add_argument('--wandb_log', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,
                help='Whether to use WANDB - this has implications for the training loop since if we want to log, we compute the metrics over training')                
    training.add_argument('--wandb_key', type=str, default=None,
                help='Path to WANDB key')    
   
    training.add_argument('--sample_size', type=int, default=64,
        help='Whether to drop UMAP/TSNE etc. for computing Higgins metric (if we do not drop them, generating the data takes ~25 hours)')      
    training.add_argument('--dataset_size', type=int, default=1000,
        help='Whether to drop UMAP/TSNE etc. for computing Higgins metric (if we do not drop them, generating the data takes ~25 hours)')      
    training.add_argument('--all_latents', type=lambda x: False if x in ["False", "false", "", "None", "0"] else True, default=0,
        help='Whether to use 5 or 4 latents in Dsprites')      
    training.add_argument('--loss-b', type=float, default=1.,
        help='beta factor for loss')
    ### Added arguments for beta annealing
    training.add_argument('--beta_start', type=float, default=1., help='Initial beta value for annealing.')
    training.add_argument('--beta_end', type=float, default=1., help='Final beta value for annealing.')
    training.add_argument('--beta_anneal_epochs', type=int, default=None, help='Number of epochs over which to anneal beta.')
    ###
    training.add_argument('--multiple_l', type=lambda x: False if x in ["False", "false", "", "None", "0"] else True, default=False,
        help='Whether to do search across L values for classifiers')

    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default='BetaVAEHiggins', choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                      default = 10,
                       help='Dimension of the latent variable.')

    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=False,
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=True,
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")

    args = parser.parse_args(args_to_parse)

    return args



def main(args):


    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    try:
        wandb_auth(dir_path=args.wandb_key)
    except:
        try:
            wandb.login()
        except:
            print(f"Authentication for WANDB failed! Trying to disable it")
            os.environ["WANDB_MODE"] = "disabled"

    # Dynamic wandb run naming
    if args.beta_start == 1.0 and args.beta_end == 1.0:
        run_name = f"StandardVAE_{args.dataset}_z={args.latent_dim}_epochs={args.epochs}"
    elif args.beta_start == args.beta_end and args.beta_start > 1.0:
        run_name = f"BetaVAE_beta={args.beta_start}_{args.dataset}_z={args.latent_dim}_epochs={args.epochs}"
    elif args.beta_start > 1.0 and args.beta_end == 1.0:
        run_name = f"AnnealedBeta_beta={args.beta_start}_to_1_{args.dataset}_z={args.latent_dim}_epochs={args.epochs}"
    else:
        run_name = f"VAE_beta={args.beta_start}_to_{args.beta_end}_{args.dataset}_z={args.latent_dim}_epochs={args.epochs}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", None),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=run_name
    )
    wandb.config.update(args)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    #set seed
    args.seed = args.seed if args.seed is not None else random.randint(1,10000)
    set_seed(args.seed)
    if args.name is None:
        args.name = args.model_type
    new_path = args.name+f"{args.seed}" if args.seed is not None else args.name
    exp_dir = os.path.join(RES_DIR, new_path)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    print("Config:")
    print(vars(args))

    # Ensure beta_anneal_epochs defaults to epochs if not set and annealing is needed
    if args.beta_anneal_epochs is None and args.beta_start != args.beta_end:
        args.beta_anneal_epochs = args.epochs

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)

        # Use train/val split from get_dataloaders
        train_loader, val_loader, dataset_obj, viz_loader, test_loader = get_dataloaders(
            args.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            val_split=0.1  # 10% for validation, adjust as needed
        )
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        logger.info("Validation set: {} samples".format(len(val_loader.dataset)))
        img_size = dataset_obj.img_size
        beta = args.loss_b

        # Set beta for model initialization: use beta_start if annealing, else loss_b
        if args.beta_anneal_epochs is not None:
            beta = args.beta_start
        else:
            beta = args.loss_b

        model = eval("{model}({latent_dim}, {beta}, {img_size}, latent_dist = 'bernoulli')".format(
            model=args.model_type, latent_dim=args.latent_dim, beta=beta, img_size=img_size))
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        scheduler = None
        if args.model_type == "BetaVAEHiggins":
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        elif args.model_type == "BetaVAEConv":
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        # elif args.model_type == "BetaVAEBurgess":
        #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
        #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=round(args.epochs*0.75), gamma=0.2)
        
        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)

        trainer = Trainer(
            model, optimizer, scheduler=scheduler, device=device, logger=logger, gif_visualizer=gif_visualizer, metrics_freq=args.metrics_freq,
            sample_size=args.sample_size, save_dir=exp_dir, dataset_size=args.dataset_size, all_latents=args.all_latents, seed=args.seed, dataset_name=args.dataset,
            beta_start=args.beta_start, beta_end=args.beta_end, beta_anneal_epochs=args.beta_anneal_epochs
        )

        # Pass val_loader to Trainer
        trainer(train_loader, epochs=args.epochs, wandb_log=args.wandb_log, val_loader=val_loader)

        latents_plots, traversal_plots, dim_reduction_models = { }, { }, { }
    
        try:
            # Remove latent_viz and dim_reduction_models (PCA/tSNE) code
            pass
        except:
            pass
    
        # Use validation loader for visualization
        viz_loader = val_loader
        viz = Visualizer(model=model,
                    model_dir=exp_dir,
                    dataset=args.dataset,
                    max_traversal=args.max_traversal,
                    loss_of_interest='val_kl_loss_',
                    upsample_factor=1)

        traversal_plots = {}
        base_datum = next(iter(train_loader))[0][0].unsqueeze(dim=0)
        # Remove traversal_plots using dim_reduction_models

        # Original plots from the repo
        size = (args.n_rows, args.n_cols)
        # same samples for all plots: sample max then take first `x`data  for all plots
        num_samples = args.n_cols * args.n_rows
        samples = get_samples(viz_loader, num_samples, idcs=args.idcs)

        if "all" in args.plots:
            args.plots = [p for p in PLOT_TYPES if p != "all"]
        builtin_plots = {}
        plot_fnames = []
        for plot_type in args.plots:
            if plot_type == 'generate-samples':
                fname, plot = viz.generate_samples(size=size)
                builtin_plots["generate-samples"] = plot
            elif plot_type == 'data-samples':
                fname, plot = viz.data_samples(samples, size=size)
                builtin_plots["data-samples"] = plot
            elif plot_type == "reconstruct":
                fname, plot = viz.reconstruct(samples, size=size)
                builtin_plots["reconstruct"] = plot
            elif plot_type == 'traversals':
                fname, plot = viz.traversals(data=samples[0:1, ...],
                        n_per_latent=args.n_cols,
                        n_latents=args.n_rows,
                        is_reorder_latents=True)
                builtin_plots["traversals"] = plot
            elif plot_type == "reconstruct-traverse":
                fname, plot = viz.reconstruct_traverse(samples,
                                        is_posterior=True,
                                        n_latents=args.n_rows,
                                        n_per_latent=args.n_cols,
                                        is_show_text=True)
                builtin_plots["reconstruct-traverse"] = plot
            elif plot_type == "gif-traversals":
                fname, plot = viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
                builtin_plots["gif-traversals"] = plot
            else:
                raise ValueError("Unkown plot_type={}".format(plot_type))
            plot_fnames.append(fname)

        converted_imgs = {}
        for k, img in builtin_plots.items():
            print(f"Converting {k}")
            try:
                converted_imgs[k] = wandb.Image(img)
            except:
                print(f"Failed to convert {k}")

        if args.wandb_log:
            wandb.log({"latents":latents_plots, "latent_traversal":traversal_plots, "builtin_plots":converted_imgs})
            for fname in plot_fnames:
                try:
                    wandb.save(fname)
                except Exception as e:
                    print(f"Failed to save {fname} to WANDB. Exception: {e}")
        

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))
        #free RAM from train loader so test loader can be loaded
        del train_loader
        gc.collect()

    if args.is_metrics:
        logger.info("Evaluation time :)")

        model.eval()
        evaluator = Evaluator(model, device=device, sample_size = args.sample_size, dataset_size = args.dataset_size, all_latents= args.all_latents, use_wandb = args.wandb_log, seed=args.seed, multiple_l=args.multiple_l)

        metrics = evaluator(test_loader, dataset_name = args.dataset)
        wandb.log({"final":{"metric":metrics}})
   
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
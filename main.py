import logging
import os

import torch

from experiments.baseline import BaselineExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from parse_args import parse_arguments


def setup_experiment(opt):
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment( opt )
        train_loader, validation_loader, test_loader = build_splits_baseline( opt )

    elif opt['experiment'] == 'domain_disentangle':

        experiment = DomainDisentangleExperiment( opt )
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle( opt )

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment( opt )
        train_loader, validation_loader, test_loader = build_splits_clip_disentangle( opt )

    else:
        raise ValueError( 'Experiment not yet supported.' )

    return experiment, train_loader, validation_loader, test_loader


def main(opt):
    experiment, train_loader, validation_loader, test_loader = setup_experiment( opt )

    if not opt['test']:  # Skip training if '--test' flag is set


        # Restore last checkpoint
        if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                f'{opt["output_path"]}/last_checkpoint.pth' )
        else:
            logging.info( opt )

        # Train loop
        for i in [0.001]:
            iteration = 0
            best_accuracy = 0
            total_train_loss = 0
            weight = torch.tensor( [7.5448e-05, 8.6793e-05, 5.4548e-07] )
            print( weight )
            while iteration < opt['max_iterations']:
                for data in train_loader:
                    if opt['experiment'] == 'domain_disentangle':
                        total_train_loss += experiment.train_iteration( data, train=True, weight=weight )

                    if iteration % opt['print_every'] == 0:
                        logging.info(
                            f'[TRAIN - {iteration} - Weight- {weight}] Loss: {total_train_loss / (iteration + 1)}' )

                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate( validation_loader )
                        logging.info(
                            f'[VAL - {iteration} Weight- {weight}]] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}' )
                        if val_accuracy > best_accuracy:
                            experiment.save_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                        best_accuracy, total_train_loss )
                        experiment.save_checkpoint( f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                    best_accuracy,
                                                    total_train_loss )

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

                if iteration <= opt['max_iterations']:
                    for data in test_loader:

                        total_train_loss += experiment.train_iteration( data, train=False, weight=weight )

                        if iteration % opt['print_every'] == 0:
                            logging.info( f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}' )

                        if iteration % opt['validate_every'] == 0:
                            # Run validation
                            val_accuracy, val_loss = experiment.validate( validation_loader )
                            logging.info(
                                f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}' )
                            if val_accuracy > best_accuracy:
                                experiment.save_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                            best_accuracy, total_train_loss )
                            experiment.save_checkpoint( f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                        best_accuracy,
                                                        total_train_loss )

                        iteration += 1
                        if iteration > opt['max_iterations']:
                            break

    # Test
    experiment.load_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth' )
    test_accuracy, _ = experiment.validate( test_loader )
    logging.info( f'[TEST] Accuracy: {(100 * test_accuracy):.2f}' )


if __name__ == '__main__':
    opt = parse_arguments()

    # Setup output directories
    os.makedirs( opt['output_path'], exist_ok=True )

    # Setup logger
    logging.basicConfig( filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO,
                         filemode='a' )

    main( opt )

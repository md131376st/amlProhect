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

        if opt['experiment'] == 'baseline':   
            
            # Restore last checkpoint
            if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
                iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                    f'{opt["output_path"]}/last_checkpoint.pth' )
            else:
                logging.info( opt ) 

            # Train loop
            iteration = 0
            best_accuracy = 0
            total_train_loss = 0
            while iteration < opt['max_iterations']:
                for data in train_loader:

                    total_train_loss += experiment.train_iteration( data )

                    if iteration % opt['print_every'] == 0:
                        logging.info(
                            f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}' )

                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate( validation_loader )
                        logging.info(
                            f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}' )
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                        best_accuracy, total_train_loss )
                        experiment.save_checkpoint( f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                    best_accuracy,
                                                    total_train_loss )

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break
        
        elif opt['experiment'] == 'domain_disentangle':  

            # Restore last checkpoint
            if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
                iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                    f'{opt["output_path"]}/last_checkpoint.pth' )
            else:
                logging.info( opt )

            # Train loop
            iteration = 0
            best_accuracy = 0
            total_train_loss = 0
            weight = torch.tensor( [1.0, 0.5, 0.3, 0.05, 0.05] )
            logging.info(
                f'WEIGHT: {weight}' )
            train_loader_iterator = iter(train_loader)
            test_loader_iterator = iter(test_loader)
            while iteration < opt['max_iterations']:

                #getting the next batch of train data
                try:
                    data = next(train_loader_iterator)
                except StopIteration:
                    train_loader_iterator = iter(train_loader)
                    data = next(train_loader_iterator)

                total_train_loss += experiment.train_iteration( data, train=True, weight=weight )

                #getting the next batch of test data
                try:
                    data = next(test_loader_iterator)
                except StopIteration:
                    test_loader_iterator = iter(test_loader)
                    data = next(test_loader_iterator)
                
                total_train_loss += experiment.train_iteration( data, train=False, weight=weight )


                if iteration % opt['print_every'] == 0:
                    logging.info( f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}' )

                if iteration % opt['validate_every'] == 0:
                    # Run validation
                    val_accuracy, val_loss = experiment.validate( validation_loader )
                    logging.info(
                        f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}' )
                    if val_accuracy >= best_accuracy:
                        best_accuracy = val_accuracy
                        experiment.save_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                    best_accuracy, total_train_loss )
                    experiment.save_checkpoint( f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                best_accuracy,
                                                total_train_loss )

                #We iterate over two batches at each iteration
                iteration += 2
                if iteration > opt['max_iterations']:
                    break
        
        elif opt['experiment'] == 'clip_disentangle':  

            # Restore last checkpoint
            if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
                iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                    f'{opt["output_path"]}/last_checkpoint.pth' )
            else:
                logging.info( opt )

            # Train loop
            iteration = 0
            best_accuracy = 0
            top5Accuracy = [0,0,0,0]
            total_train_loss = 0
            counter =0

            weight = torch.tensor( [1.0, 0.5, 0.3, 0.05, 0.05, 0.2] )
            logging.info(
                f'WEIGHT: {weight}' )
            train_loader_iterator = iter(train_loader)
            test_loader_iterator = iter(test_loader)
            while iteration < opt['max_iterations']:
                
                #getting the next batch of train data
                try:
                    data = next(train_loader_iterator)
                except StopIteration:
                    train_loader_iterator = iter(train_loader)
                    data = next(train_loader_iterator)

                total_train_loss += experiment.train_iteration( data, train=True, weight=weight )

                #getting the next batch of test data
                try:
                    data = next(test_loader_iterator)
                except StopIteration:
                    test_loader_iterator = iter(test_loader)
                    data = next(test_loader_iterator)
                
                total_train_loss += experiment.train_iteration( data, train=False, weight=weight )


                if iteration % opt['print_every'] == 0:
                    logging.info( f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}' )

                if iteration % opt['validate_every'] == 0:
                    # Run validation
                    val_accuracy, val_loss = experiment.validate( validation_loader )
                    logging.info(
                        f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}' )
                    """
                    1)In this section we compare best value with the accuracy on the validation set.
                    2)In the case of better value, we remove the first best value from the queue of top5accuracy.
                    3)We save the the last best value in the queue. 
                    4)rename the files remained to point to the correct accuracy.
                    5)We change the best_accuracy value to the current accuracy on the validation set.
                    """
                    if val_accuracy >= best_accuracy:
                        top5Accuracy.pop(0)
                        top5Accuracy.append(best_accuracy)
                        if os.path.isfile(f'{opt["output_path"]}/best1_checkpoint.pth'):
                            os.remove(f'{opt["output_path"]}/best1_checkpoint.pth')
                        for i in range(3):
                            if os.path.isfile(f'{opt["output_path"]}/best{i+2}_checkpoint.pth'):
                                os.rename(f'{opt["output_path"]}/best{i+2}_checkpoint.pth', f'{opt["output_path"]}/best{i+1}_checkpoint.pth')
                        experiment.save_checkpoint(f'{opt["output_path"]}/best4_checkpoint.pth', iteration,
                                                   best_accuracy, total_train_loss)

                        best_accuracy = val_accuracy
                        experiment.save_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                    best_accuracy, total_train_loss )

                    experiment.save_checkpoint( f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                best_accuracy,
                                                total_train_loss )
                    if iteration % 1000== 0:
                        experiment.save_checkpoint(f'{opt["output_path"]}/{counter}_checkpoint.pth', iteration,
                                                   best_accuracy,
                                                   total_train_loss)
                        counter = counter +1

                #We iterate over two batches at each iteration
                iteration += 2
                if iteration > opt['max_iterations']:
                    logging.info(best_accuracy)
                    logging.info(top5Accuracy)
                    break
        

    # Test
    """
    1) we test the models base  best validation
    2) if the experiment is clip_disentangle we try to check accuracy in every 1000 iterations, 
    best4latest validation, the last run
    """
    experiment.load_checkpoint( f'{opt["output_path"]}/best_checkpoint.pth' )
    test_accuracy, _ = experiment.validate( test_loader )
    logging.info( f'[TEST] Accuracy best: {(100 * test_accuracy):.2f}' )
    if opt['experiment'] == 'clip_disentangle':
        experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        test_accuracy, _ = experiment.validate(test_loader)
        logging.info(f'[TEST] Accuracy last: {(100 * test_accuracy):.2f}')
        for i in range(4):
            if os.path.isfile(f'{opt["output_path"]}/best{i + 1}_checkpoint.pth'):
                experiment.load_checkpoint(f'{opt["output_path"]}/best{i + 1}_checkpoint.pth')
                test_accuracy, _ = experiment.validate(test_loader)
                logging.info(f'[TEST] Accuracy best {i}: {(100 * test_accuracy):.2f}')
        for i in range(int(opt['max_iterations']/1000)):
            if os.path.isfile(f'{opt["output_path"]}/{i}_checkpoint.pth'):
                experiment.load_checkpoint(f'{opt["output_path"]}/{i}_checkpoint.pth')
                test_accuracy, _ = experiment.validate(test_loader)
                logging.info(f'[TEST] Accuracy count {i}: {(100 * test_accuracy):.2f}')


if __name__ == '__main__':
    opt = parse_arguments()

    # Setup output directories
    os.makedirs( opt['output_path'], exist_ok=True )

    # Setup logger
    logging.basicConfig( filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO,
                         filemode='a' )

    main( opt )

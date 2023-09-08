from argparse import ArgumentParser
import os
from pathlib import Path
import time
import tqdm
import torch

from solver.umx import UMXSolver
from solver.spl_cac import SPLCaCSolver
from solver.spl import SPLSolver
from solver.demucs import DemucsSolver
from utils import read_config_yaml, EarlyStopping


def check_args(model_dir: str,
               config_path: str,
               dataset_dir: str,
               train_folder_name: str,
               val_folder_name: str):
    if model_dir is not None:
        if not os.path.isdir(model_dir):
            raise FileExistsError(f'model_dir {model_dir} does not exist.')

    if not os.path.isfile(config_path):
        raise FileExistsError(f'config_path {config_path} does not exist.')

    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(f'dataset_dir {dataset_dir} does not exist.')

    if not os.path.isdir(os.path.join(dataset_dir, train_folder_name)):
        raise NotADirectoryError(f'train_dir {os.path.join(dataset_dir, train_folder_name)} does not exist.')

    if not os.path.isdir(os.path.join(dataset_dir, val_folder_name)):
        raise NotADirectoryError(f'val_dir {os.path.join(dataset_dir, val_folder_name)} does not exist.')


def print_model_size(solver):
    num_params = sum(p.numel() for p in solver._model.parameters())
    mb = num_params * 4 / 2 ** 20
    print(f'Num params: {num_params}, Model Size: {mb} MB.')


if __name__ == "__main__":
    parser = ArgumentParser('Specify your model to train and its configurations.')
    parser.add_argument('-b', '--batch_size', type=int, default=None)
    parser.add_argument('-c', '--config_path', type=str, default=None)
    parser.add_argument('-d', '--dataset_dir', type=str, default='')
    parser.add_argument('-m', '--model', type=str, default='umx')
    parser.add_argument('-M', '--model_dir', type=str, default=None)
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-t', '--train_folder_name', type=str, default='train_aug')
    parser.add_argument('-v', '--val_folder_name', type=str, default='val_aug')
    parser.add_argument('--cpu', dest='cpu', action='store_true', default=False)
    parser.add_argument('--quiet', dest='quiet', action='store_true', default=False)

    args = parser.parse_args()
    model_dir = args.model_dir
    config_path = args.config_path
    dataset_dir = args.dataset_dir
    model = args.model
    num_workers = args.num_workers
    output_dir = args.output_dir
    train_folder_name = args.train_folder_name
    val_folder_name = args.val_folder_name
    cpu = args.cpu
    quiet = args.quiet

    use_cuda = False if cpu else True
    device = torch.device("cuda" if use_cuda else "cpu")

    model_cfg = read_config_yaml(config_path)

    if args.batch_size is not None:
        model_cfg.train.batch_size = args.batch_size

    # Check arguments
    check_args(model_dir, config_path, dataset_dir, train_folder_name, val_folder_name)

    # Create directories for the output paths
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
    es = EarlyStopping(patience=model_cfg.train.patience)

    if model == 'umx':
        umx_solver = UMXSolver(device=device,
                               model_cfg=model_cfg,
                               dataset_dir=dataset_dir,
                               output_dir=output_dir,
                               train_folder_name=train_folder_name,
                               val_folder_name=val_folder_name,
                               dataloader_kwargs=dataloader_kwargs)

        print_model_size(umx_solver)

        if model_dir is not None:
            umx_solver.load_checkpoint(model_dir)
            results = umx_solver.load_params(model_dir)

        t = tqdm.trange(1, model_cfg.train.num_epochs + 1, disable=quiet)

        for epoch in t:
            t.set_description("Training epoch")
            end = time.time()
            train_loss = umx_solver.train(epoch=epoch)
            val_loss = umx_solver.val()

            t.set_postfix(train_loss=train_loss,
                          val_loss=val_loss)

            stop = umx_solver.es.step(val_loss)

            if val_loss == umx_solver.es.best:
                umx_solver.best_epoch = epoch

            umx_solver.save_checkpoint(epoch=epoch + 1,
                                       best_loss=umx_solver.es.best,
                                       is_best=(val_loss == umx_solver.es.best))

            umx_solver.save_params(params={'epochs_trained': epoch + 1,
                                           'best_loss': umx_solver.es.best,
                                           'best_epoch': umx_solver.best_epoch,
                                           'train_loss_history': umx_solver.train_losses,
                                           'val_loss_history': umx_solver.val_losses,
                                           'train_time_history': umx_solver.train_times,
                                           'num_bad_epochs': umx_solver.es.num_bad_epochs})

            umx_solver.train_times.append(time.time() - end)

            if stop:
                print("Apply Early Stopping")
                break

    elif model == 'spl':
        spl_solver = SPLSolver(device=device,
                               model_cfg=model_cfg,
                               dataset_dir=dataset_dir,
                               output_dir=output_dir,
                               train_folder_name=train_folder_name,
                               val_folder_name=val_folder_name,
                               dataloader_kwargs=dataloader_kwargs)

        t = tqdm.trange(1, model_cfg.train.num_epochs + 1, disable=quiet)
        best_epoch = 0

        if model_dir is not None:
            spl_solver.load_checkpoint(model_dir)
            results = spl_solver.load_params(model_dir)

        print_model_size(spl_solver)

        for epoch in t:
            t.set_description("Training epoch")
            end = time.time()
            train_loss = spl_solver.train(epoch=epoch)
            val_loss = spl_solver.val()

            t.set_postfix(train_loss=train_loss,
                          val_loss=val_loss)

            stop = spl_solver.es.step(val_loss)

            if val_loss == spl_solver.es.best:
                spl_solver.best_epoch = epoch

            spl_solver.save_checkpoint(epoch=epoch + 1,
                                       best_loss=spl_solver.es.best,
                                       is_best=(val_loss == spl_solver.es.best))

            spl_solver.save_params(params={'epochs_trained': epoch + 1,
                                           'best_loss': spl_solver.es.best,
                                           'best_epoch': spl_solver.best_epoch,
                                           'train_loss_history': spl_solver.train_losses,
                                           'val_loss_history': spl_solver.val_losses,
                                           'train_time_history': spl_solver.train_times,
                                           'num_bad_epochs': spl_solver.es.num_bad_epochs})

            spl_solver.train_times.append(time.time() - end)

            if stop:
                print("Apply Early Stopping")
                break

    elif model == 'spl_cac':
        spl_solver = SPLCaCSolver(device=device,
                                  model_cfg=model_cfg,
                                  dataset_dir=dataset_dir,
                                  output_dir=output_dir,
                                  train_folder_name=train_folder_name,
                                  val_folder_name=val_folder_name,
                                  dataloader_kwargs=dataloader_kwargs)

        t = tqdm.trange(1, model_cfg.train.num_epochs + 1, disable=quiet)
        best_epoch = 0

        if model_dir is not None:
            spl_solver.load_checkpoint(model_dir)
            results = spl_solver.load_params(model_dir)

        print_model_size(spl_solver)

        for epoch in t:
            t.set_description("Training epoch")
            end = time.time()
            train_loss = spl_solver.train(epoch=epoch)
            val_loss = spl_solver.val()

            t.set_postfix(train_loss=train_loss,
                          val_loss=val_loss)

            stop = spl_solver.es.step(val_loss)

            if val_loss == spl_solver.es.best:
                spl_solver.best_epoch = epoch

            spl_solver.save_checkpoint(epoch=epoch + 1,
                                       best_loss=spl_solver.es.best,
                                       is_best=(val_loss == spl_solver.es.best))

            spl_solver.save_params(params={'epochs_trained': epoch + 1,
                                           'best_loss': spl_solver.es.best,
                                           'best_epoch': spl_solver.best_epoch,
                                           'train_loss_history': spl_solver.train_losses,
                                           'val_loss_history': spl_solver.val_losses,
                                           'train_time_history': spl_solver.train_times,
                                           'num_bad_epochs': spl_solver.es.num_bad_epochs})

            spl_solver.train_times.append(time.time() - end)

            if stop:
                print("Apply Early Stopping")
                break

    elif model == 'demucs':
        demucs_solver = DemucsSolver(device=device,
                                     model_cfg=model_cfg,
                                     dataset_dir=dataset_dir,
                                     output_dir=output_dir,
                                     train_folder_name=train_folder_name,
                                     val_folder_name=val_folder_name,
                                     dataloader_kwargs=dataloader_kwargs)

        t = tqdm.trange(1, model_cfg.train.num_epochs + 1, disable=quiet)
        train_times = []
        best_epoch = 0

        if model_dir is not None:
            demucs_solver.load_checkpoint(model_dir)
            results = demucs_solver.load_params(model_dir)

        print_model_size(demucs_solver)
        test_metric = demucs_solver.test_metric

        for epoch in t:
            t.set_description("Training epoch")
            end = time.time()
            train_losses = demucs_solver.train(epoch=epoch)
            val_losses = demucs_solver.val()

            t.set_postfix(train_loss=train_losses[test_metric],
                          val_loss=val_losses[test_metric])

            stop = es.step(val_losses[test_metric])

            if val_losses[test_metric] == es.best:
                demucs_solver.best_epoch = epoch

            demucs_solver.save_checkpoint(epoch=epoch + 1,
                                          best_loss=es.best,
                                          is_best=(val_losses[test_metric] == es.best))

            demucs_solver.save_params(params={'epochs_trained': epoch + 1,
                                              'best_loss': demucs_solver.es.best,
                                              'best_epoch': demucs_solver.best_epoch,
                                              'train_loss_history': demucs_solver.train_losses,
                                              'val_loss_history': demucs_solver.val_losses,
                                              'train_time_history': demucs_solver.train_times,
                                              'num_bad_epochs': es.num_bad_epochs})

            train_times.append(time.time() - end)

            if stop:
                print("Apply Early Stopping")
                break

    else:
        NotImplementedError()




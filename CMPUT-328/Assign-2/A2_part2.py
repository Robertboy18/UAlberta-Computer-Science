import torch
import timeit
from pprint import pformat

from A2_submission import get_dataset, tune_hyper_parameter

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Running on CPU')

    dataloaders, params = get_dataset('CIFAR10')

    start = timeit.default_timer()

    tune_hyper_parameter(dataloaders, device, params)

    stop = timeit.default_timer()
    run_time = stop - start

    print("\nrun_time:\n", pformat(run_time))


if __name__ == '__main__':
    main()


import torch
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat

from A2_submission_soln import LogisticRegression, get_dataset, train, test

# from A2_submission import LogisticRegression, get_dataset, train, test


torch.multiprocessing.set_sharing_strategy('file_system')


def compute_score(acc, run_time, min_thres, max_thres, max_run_time):
    if run_time > max_run_time:
        return 0.0

    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def run(dataset_name, device):
    dataloaders, params = get_dataset(dataset_name)

    start = timeit.default_timer()

    model = LogisticRegression(params).to(device)

    train(model, dataloaders['train'], dataloaders['valid'], device, params)

    predicted_test_labels, gt_labels = test(model, dataloaders['test'], device, params)

    if predicted_test_labels is None or gt_labels is None:
        return 0, 0, 0

    stop = timeit.default_timer()
    run_time = stop - start

    # np.savetxt(filename, np.asarray(predicted_test_labels))

    correct = 0
    total = 0
    for label, prediction in zip(gt_labels, predicted_test_labels):
        total += label.size(0)
        correct += (prediction.cpu().numpy() == label.cpu().numpy()).sum().item()  # assuming your model runs on GPU

    accuracy = float(correct) / total

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct, accuracy, run_time


"""Main loop. Run time and total score will be shown below."""


def run_on_dataset(dataset_name, device):
    max_run_time = 200
    if dataset_name == "MNIST":
        min_thres = 0.82
        max_thres = 0.92

    elif dataset_name == "CIFAR10":
        min_thres = 0.28
        max_thres = 0.38

    correct_predict, accuracy, run_time = run(dataset_name, device)

    score = compute_score(accuracy, run_time, min_thres, max_thres, max_run_time)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    return result, score


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Running on CPU')

    result_all = OrderedDict()
    score_weights = [0.5, 0.5]
    scores = []
    for dataset_name in ["MNIST", "CIFAR10"]:
        result_all[dataset_name], this_score = run_on_dataset(dataset_name, device)
        scores.append(this_score)
    total_score = [score * weight for score, weight in zip(scores, score_weights)]
    total_score = np.asarray(total_score).sum().item()
    result_all['total_score'] = total_score
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
from common.utils import is_approx, vec_unit_dot


def training_plots(h, args, loss_type, plot_path):
    h_epoch = np.array(h['h_epoch'])
    h_acc_test = np.array(h['h_acc_test'])
    h_acc_train = np.array(h['h_acc_train'])
    h_acc_train_std = np.array(h['h_acc_train_std'])
    h_loss_test = np.array(h['h_loss_test'])
    h_loss_train = np.array(h['h_loss_train'])
    h_loss_train_std = np.array(h['h_loss_train_std'])
    h_uplink = np.array(h['h_uplink'])
    # h_downlink = np.array(h['h_downlink'])
    h_grad = h['h_grad']
    # h_grad_norm = np.array(h['h_grad_norm'])
    h_error = np.array(h['h_error'])

    grad_wrt_first = []
    grad_wrt_prev = []
    grad1 = [_.flatten() for _ in h_grad[0]]
    for idx, grad in enumerate(h_grad[1:], 1):
        gradi = [_.flatten() for _ in grad]
        gradi_minus_1 = [_.flatten() for _ in h_grad[idx-1]]
        grad_wrt_first.append([
            vec_unit_dot(one, two)
            for one, two in zip(grad1, gradi)])
        grad_wrt_prev.append([
            vec_unit_dot(one, two)
            for one, two in zip(gradi_minus_1, gradi)])
    grad_wrt_first = np.array(grad_wrt_first)
    grad_wrt_prev = np.array(grad_wrt_prev)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(221)
    ax1_ = ax1.twinx()
    l1 = ax1.plot(h_epoch, h_acc_test, 'b', label='test acc')
    l1_ = ax1.plot(h_epoch, h_acc_train, 'b.-.', label='train acc')
    ax1.fill_between(h_epoch, h_acc_train-h_acc_train_std,
                     h_acc_train+h_acc_train_std, alpha=0.3, facecolor='b')
    l2 = ax1_.plot(h_epoch, h_loss_test, 'r', label='test loss')
    l2_ = ax1_.plot(h_epoch, h_loss_train, 'r.-.', label='train loss')
    ax1_.fill_between(h_epoch, h_loss_train-h_loss_train_std,
                      h_loss_train+h_loss_train_std, alpha=0.3, facecolor='r')
    ls = l1+l1_+l2+l2_
    lab = [_.get_label() for _ in ls]
    ax1.legend(ls, lab, loc=7)
    ax1.set_ylabel('accuracy')
    ax1_.set_ylabel('{} loss'.format(loss_type))
    ax1.set_xlabel('epochs')
    ax1.set_title('(a)', y=-0.3)

    ax2 = fig.add_subplot(222)
    if is_approx(args) or (args.paradigm and 'topk' in args.paradigm):
        ax2.plot(h_epoch, h_error, 'r')
        ax2.set_ylabel('error in projection')
    # else:
    #     for col in range(h_grad_norm.shape[1]):
    #         ax2.plot(h_epoch, h_grad_norm[:, col])
    #     ax2.set_ylabel('grad norm')
    ax2.set_xlabel('epochs')
    ax2.set_title('(b)', y=-0.3)

    ax3 = fig.add_subplot(223)
    type_ = 'W'
    for col in range(grad_wrt_first.shape[1]):
        ax3.plot(h_epoch[1:], grad_wrt_first[:, col],
                 label='wrt first (L{}:{})'.format(col+1, type_))
        ax3.plot(h_epoch[1:], grad_wrt_prev[:, col],
                 label='wrt previous (L{}:{})'.format(col+1, type_))
        if type_ == 'W':
            type_ = 'B'
        elif type_ == 'B':
            type_ = 'W'
    ax3.set_ylabel('unit dot'.format(loss_type))
    ax3.set_xlabel('epochs')
    if grad_wrt_first.shape[1] < 5:
        ax3.legend()
    ax3.set_title('(c)', y=-0.3)

    ax4 = fig.add_subplot(224)
    ax5 = ax4.twinx()
    l4 = ax4.plot(h_epoch, np.cumsum(h_uplink), 'r', label='uplink')
    # l5 = ax5.plot(h_epoch, np.cumsum(h_downlink), 'g', label='downlink')
    ax4.set_ylabel('# uplink')
    ax5.set_ylabel('# downlink')
    ax4.set_xlabel('epochs')
    ls = l4  # +l5
    lab = [_.get_label() for _ in ls]
    ax4.legend(ls, lab, loc=7)
    ax4.set_title('(d)', y=-0.3)

    ax1.grid()
    ax3.grid()
    ax2.grid()
    ax4.grid()
    plt.xlim(left=0, right=args.epochs)
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    plt.savefig(plot_path, bbox_inches='tight', dpi=100)
    print('Saved: ', plot_path)

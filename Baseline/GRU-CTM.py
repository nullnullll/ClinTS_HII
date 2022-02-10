import argparse
import time
import utils
import models
import torch
import torch.nn as nn
import numpy as np
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rec-hidden', type=int, default=128)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--num-heads', type=int, default=4)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--linear-combine', action='store_true')
parser.add_argument('--alpha', type=int, default=5.)
parser.add_argument('--with-treatment', action='store_true')
parser.add_argument('--causal-masking', action='store_true')
parser.add_argument('--sample-times', type=int, default=3)
parser.add_argument('--least-winsize', type=float, default=0.8)
parser.add_argument('--masked-ratio', type=float, default=0.5)
parser.add_argument('--early-stop', type=int, default=15)
parser.add_argument('--task', type=str, default='in_hospital_mortality')
parser.add_argument('--cip', type=str, default='vaso')
parser.add_argument('--withoutheter', action='store_true')
parser.add_argument('--withoutirr', action='store_true')

args = parser.parse_args()
args.niters=10
args.lr=0.0001
args.alpha=5
args.batch_size=32
args.rec_hidden=128
args.save=1
args.classif=True
args.num_heads=4
args.learn_emb=True
args.dataset = 'mimiciii'
args.seed=0
args.with_treatment = True
args.sample_times = 3
args.task = 'in_hospital_mortality'


if __name__ == '__main__':

    seed = args.seed
    np.random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(args.task)
    if args.task == 'in_hospital_mortality':
        data_obj = utils.get_mimiciii_data(args)
    elif args.task == 'decom':
        data_obj = utils.get_decom_data(args)
    elif args.task == 'cip':
        data_obj = utils.get_cip_data(args)
    elif args.task == 'wbm':
        data_obj = utils.get_wbm_data(args)
    elif args.task == 'los':
        data_obj = utils.get_los_data(args)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    vitals_dim = 12
    event_dim =13

    # model

    rec = models.make_Encoder_GRU(args, dim).to(device)
    dec = models.make_Decoder_GRU(args, vitals_dim).to(device)
    dec_point_process = models.make_Decoder_GRU(args,event_dim).to(device)
    if args.task == 'cip':
        classifier = models.Classifier(args.rec_hidden * 2, 128, N=4, args=args).to(device)
    elif args.task == 'wbm':
        classifier = models.Classifier(args.rec_hidden * 2, 128, N=12, args=args).to(device)
    elif args.task == 'los':
        classifier = models.Classifier(args.rec_hidden * 2, 128, N=9, args=args).to(device)
    else:
        classifier = models.Classifier(args.rec_hidden * 2, 128, args=args).to(device)


    params = (list(rec.parameters())  + list(dec.parameters()) + list(dec_point_process.parameters()) +list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))

    optimizer = optim.Adam(params, lr=args.lr)
    if args.task == 'wbm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    name , save_path = utils.getFileNameandPath(args)

    print(args)
    best_val_auroc = float(0)
    total_time = 0.
    times_of_keep_unchange = 0
    stop = False
    for itr in range(1, args.niters + 1):
        if stop == True:
            break
        train_autoregressive_loss, train_time_loss, train_ce_loss = 0, 0, 0
        train_loss = 0
        train_n = 0
        train_acc = 0
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            observed_data, observed_mask, observed_tp = \
                train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]

            if args.with_treatment:
                if args.causal_masking:
                    causal_mask = utils.get_causal_mask(observed_mask, observed_tp, dim, device)
                    causal_mask = causal_mask[:, :, :, 12:25]
                else:
                    causal_mask = None
            else:
                causal_mask = None

            ClsInput = rec(torch.cat((observed_data, observed_mask), 2), observed_tp, causal_mask=causal_mask)
            pred_y = classifier(ClsInput, args)
            if args.task == 'wbm':
                ce_loss = criterion(pred_y, label.float())
            else:
                ce_loss = criterion(pred_y, label)


            prediction_timestamps = dec_point_process(ClsInput, observed_tp)
            time_stamps = observed_tp.clone()
            mask = (observed_mask[:, :, vitals_dim:] != 0).to(device)
            time_stamps = time_stamps.unsqueeze(-1).repeat_interleave(event_dim, dim=-1)
            scale_time_loss = 100
            time_loss = utils.calculate_time_loss(prediction_timestamps,time_stamps,mask,device) / scale_time_loss

            autoregressive_loss=torch.tensor(0.).to(device)
            for i in range(args.sample_times):
                if observed_mask.sum() == 0:
                    break
                while True:
                    window_size = int(torch.rand(1) * observed_data.shape[1] * (1 - args.least_winsize) \
                                      + observed_data.shape[1] * args.least_winsize)
                    if observed_mask[:, (observed_data.shape[1] - window_size):, :vitals_dim].sum() != 0:
                        break
                observation_window = observed_tp[:, :window_size]
                forecast_window = observed_tp[:, (observed_data.shape[1] - window_size):]
                if args.with_treatment:
                    if args.causal_masking:
                        ar_causal_mask = causal_mask[:, :, :window_size, :]
                    else:
                        ar_causal_mask = None
                else:
                    ar_causal_mask = None
                ENCoutput = rec(torch.cat((observed_data[:, :window_size, :].to(device), observed_mask[:, :window_size, :].to(device)),2),
                                observation_window.to(device), causal_mask=ar_causal_mask)
                out = dec(ENCoutput, forecast_window.to(device))
                mask = (observed_mask[:, (observed_data.shape[1] - window_size):, :vitals_dim] != 0).to(device)
                autoregressive_loss += utils.mean_squared_error(out[mask], observed_data[:, (observed_data.shape[1] - window_size):, :vitals_dim][mask])

            loss = autoregressive_loss / args.sample_times + time_loss + args.alpha * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_ce_loss += ce_loss.item() * batch_len
            train_autoregressive_loss += autoregressive_loss.item() * batch_len
            train_time_loss += time_loss.item() * batch_len
            train_n += batch_len

        total_time += time.time() - start_time
        if args.task == 'cip' or args.task == 'los':
            val_loss, val_acc, val_auroc, val_auroc_split, val_auprc = utils.evaluate_classifier_cip(rec, dec, val_loader,args=args,classifier=classifier,dim=dim)
            test_loss, test_acc, test_auroc, test_auroc_split, test_auprc = utils.evaluate_classifier_cip(rec, dec, test_loader,args=args,classifier=classifier,dim=dim)
        elif args.task == 'wbm':
            val_loss, val_acc, val_auroc, val_auroc_split, val_auprc = utils.evaluate_classifier_wbm(rec, dec,val_loader,args=args,classifier=classifier,dim=dim)
            test_loss, test_acc, test_auroc, test_auroc_split, test_auprc = utils.evaluate_classifier_wbm(rec, dec,test_loader,args=args,classifier=classifier,dim=dim)
        else:
            val_loss, val_acc, val_auroc, val_auprc = utils.evaluate_classifier(rec, dec, val_loader, args=args,classifier=classifier, dim=dim)
            test_loss, test_acc, test_auroc, test_auprc = utils.evaluate_classifier(rec, dec, test_loader, args=args,classifier=classifier, dim=dim)
        message ='Iter: {}, AR_loss: {:.4f}, time_loss: {:.4f}, ce_loss: {:.4f},  val_loss: {:.4f}, val_acc: {:.4f}, val_auroc_macro: {:.4f}, val_auprc: {:.4f}, test_acc: {:.4f}, test_auroc_macro: {:.4f}, test_auprc: {:.4f}' \
            .format(itr, train_autoregressive_loss/train_n, train_time_loss/train_n, train_ce_loss / train_n,  val_loss, val_acc, val_auroc, val_auprc, test_acc,
                    test_auroc, test_auprc)
        print(message)
        if val_auroc >= best_val_auroc:
            times_of_keep_unchange = 0
            best_val_auroc = max(best_val_auroc, val_auroc)
            best_test_auroc = test_auroc
            best_test_auprc = test_auprc
            if args.task == 'cip' or args.task == 'wbm' or args.task == 'los' :
                best_test_auroc_split = test_auroc_split
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            dec_point_process_dict = dec_point_process.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        else:
            times_of_keep_unchange += 1

        if times_of_keep_unchange > args.early_stop:
            stop = True
        if itr % 25 == 0 and itr !=200 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'dec_point_process_dict':dec_point_process_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, '../checkpoint' + args.task + '/' + 'GRU-CTM-irr' + '_' + str(args.seed) + '.h5')

    print(best_test_auroc)
    print(best_test_auprc)
    if args.task == 'cip' or args.task == 'wbm' or args.task == 'los':
        print(best_test_auroc_split)


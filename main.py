import os
import time
import traceback

import pandas as pd
import torch
from torch.utils.data import DataLoader

from conf import batch_size
from detect import detect
from train import train_stage1, train_stage2
from utils.dataset import Dataset
import torch.utils.data as Data

from utils.eval import cal_best_PRF


def main(dataset):
    '''
    :param dataset:
    :return:
    '''


    Xs=[]
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append( torch.LongTensor(dataset.features[i]))
    mask=torch.BoolTensor(dataset.mask)
    train_Dataset = Data.TensorDataset(*Xs, mask)
    detect_Dataset = Data.TensorDataset(*Xs, mask)

    train_dataloader = DataLoader(train_Dataset, batch_size,shuffle=True,num_workers=0,pin_memory=True, drop_last=True)

    encoder, decoder = train_stage1(train_dataloader, dataset.attribute_dims, dataset.max_len)
    encoder, decoder = train_stage2(train_dataloader, encoder, decoder, dataset.attribute_dims)

    detect_dataloader = DataLoader(detect_Dataset, batch_size,
                            shuffle=False,num_workers=0,pin_memory=True)

    attr_Shape=(dataset.num_cases,dataset.max_len,dataset.num_attributes)
    trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores = detect(encoder,decoder, detect_dataloader, dataset.attribute_dims,attr_Shape=attr_Shape)

    return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores





if __name__ == '__main__':
    filePath = 'eventlogs'
    resPath='result.csv'
    dataset_names = os.listdir(filePath)
    dataset_names.sort()
    dataset_names.remove('cache')

    for dataset_name in dataset_names:
        try:
            print(dataset_name)
            start_time = time.time()
            dataset = Dataset(dataset_name)
            trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores= main(dataset)

            end_time = time.time()

            run_time = end_time - start_time
            print(run_time)

            ##trace level
            trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
            print("Trace-level anomaly detection")
            print(f'precision: {trace_p}, recall: {trace_r}, F1-score: {trace_f1}, AP: {trace_aupr}')  #AP：average precision score

            ##event level
            eventTemp = dataset.binary_targets.sum(2).flatten()
            eventTemp[eventTemp > 1] = 1
            event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
            print("Event-level anomaly detection")
            print(f'precision: {event_p}, recall: {event_r}, F1-score: {event_f1}, AP: {event_aupr}')

            ##attr level
            attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                              attr_level_abnormal_scores.flatten())

            print("Attribute-level anomaly detection")
            print(f'precision: {attr_p}, recall: {attr_r}, F1-score: {attr_f1}, AP: {attr_aupr}')


            datanew = pd.DataFrame([{'index': dataset_name, 'trace_p': trace_p, "trace_r": trace_r, 'trace_f1': trace_f1,
                      'trace_AP': trace_aupr,
                      'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_AP': event_aupr,
                      'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_AP': attr_aupr
                                     }])
            if os.path.exists(resPath):
                data = pd.read_csv(resPath)
                data = data.append(datanew, ignore_index=True)
            else:
                data = datanew
            data.to_csv(resPath, index=False)
        except:
            traceback.print_exc()
            datanew = pd.DataFrame([{'index': dataset_name}])
            if os.path.exists(resPath):
                data = pd.read_csv(resPath)
                data = data.append(datanew, ignore_index=True)
            else:
                data = datanew
            data.to_csv(resPath, index=False)

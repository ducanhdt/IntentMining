import numpy as np
import itertools
import pandas as pd
import os
from evaluation import EvaluateDataset
import glob

DATA_PATH = "data/"
RESULT_PATH = "result/"

def generate_parameters_dbscan():
    param_grid = {
        "min_distance": np.arange(0.09, 0.40, 0.01),
        "minimum_samples": range(3, 20)
    }
    keys, values = zip(*param_grid.items())

    all_parameters = []
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        all_parameters.append(hyperparameters)

    # all_parameters = all_parameters[:5]
    print("Parameter search space size: " + str(len(all_parameters)))
    return all_parameters


def generate_parameters_hdbscan():
    param_grid = {
        "min_cluster_size": range(3, 15),
        "min_samples": range(3, 15)
    }
    keys, values = zip(*param_grid.items())

    all_parameters = []
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        all_parameters.append(hyperparameters)

    # all_parameters = all_parameters[:5]
    print("Parameter search space size: " + str(len(all_parameters)))
    return all_parameters


def generate_parameters_iter_dbscan():
    param_grid = {
        "distance": [0.03, 0.15, 0.3],
        "max_iteration": list(range(10, 20, 3)),
        "minimum_samples": list(range(8, 20,3))
    }
    keys, values = zip(*param_grid.items())
    all_parameters = []
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        all_parameters.append(hyperparameters)

    # all_parameters = all_parameters[:5]
    print("Parameter search space size: " + str(len(all_parameters)))
    return all_parameters

def generate_combined_result(corpus, directory) :
    experiments = glob.glob(directory + '/*')
    df = pd.read_csv(DATA_PATH + corpus + '.csv')
    total_intents = len(df['label'].value_counts())
    output = []
    for exp in experiments:
        df = pd.read_excel(exp)
        if "_parameters.xlsx" not in exp: continue
        algo_name = exp.split('/')[-1].replace('_parameters.xlsx', '')
        try:
            nmi = round(np.mean(df['normalized_mutual_info_score']), 2)
        except:
            nmi = 0.0

        try :
            max_nmi = np.max(df['normalized_mutual_info_score'].values)
        except :
            max_nmi = 0.0
        try:
            ars = round(np.mean(df['adjusted_rand_score']), 2)
        except:
            ars = 0.0
        try :
            max_ars = np.max(df['adjusted_rand_score'].values)
        except:
            max_ars = 0.0

        try:
            acc = round(np.mean(df['accuracy']), 2)
        except:
            acc = 0.0
        try :
            max_acc = np.max(df['accuracy'].values)
        except:
            max_acc = 0.0

        try :
            f1 = np.mean(df['f1'].values)
        except:
            f1 = 0.0

        try :
            max_f1 = np.max(df['f1'].values)
        except:
            max_f1 = 0.0

        try:
            intents = max(df['intents'].values.tolist())
        except:
            intents = 0
        try:
            avg_clusters = int(np.mean(df['clusters']))
        except:
            avg_clusters = 0
        try:
            min_clusters = min(df.loc[df['intents'] == intents]['clusters'].values.tolist())
        except:
            min_clusters = 0

        output.append([corpus, algo_name, total_intents, intents, min_clusters, avg_clusters, nmi, max_nmi, ars, max_ars,
                       acc, max_acc, f1, max_f1])
    return output

if __name__ == '__main__':
    # corpus_lists = {'Airlines', 'AskUbuntuCorpus', 'ChatbotCorpus', 'WebApplicationsCorpus',
    #                'FinanceData', 'ATIS', 'PersonalAssistant'}
    corpus_lists = [
        "61139b4729ef9b70788ff84c", # : TEST DATA - VCB Cũ
        "6114ca6f8c9d2ad4f57802b3", # : TLA
        "6114cb708c9d2ac9d278345a", # : VCB mới
        "6113a52429ef9bd2aa9086db", # : TEST DATA - POC VIETINBANK
        "6113a30e29ef9bfc53907c6b", # : TEST DATA - VOICE BOT BIG
        "6113a71729ef9b297690a5b3", # : TEST DATA - VOICEBOT Sacombank
]

    algorithms = ['ITER_DBSCAN']

    for c in corpus_lists:
        print("Processing corpus ...", c)
        evaluate = EvaluateDataset(DATA_PATH + c + '.csv', filetype='csv', text_column='text', target_column='label')
        for algo in algorithms:
            param_results = None
            if algo == 'DBSCAN':
                parameters = generate_parameters_dbscan()
                param_results = evaluate.run_iter(parameters, algo)
                print("Total number of result: ", len(param_results))
            elif algo == 'HDBSCAN':
                parameters = generate_parameters_hdbscan()
                param_results = evaluate.run_iter(parameters, algo)
                print("Total number of result: ", len(param_results))
            elif algo == 'ITER_DBSCAN':
                parameters = generate_parameters_iter_dbscan()
                param_results = evaluate.run_iter(parameters, algo)
                print("Total number of result: ", len(param_results))

            if param_results is not None:
                parameter_df = pd.DataFrame.from_dict(param_results)


                directory = RESULT_PATH + c + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                parameter_df.to_excel(directory + algo + '_parameters.xlsx', index=False)
            output = generate_combined_result(c, RESULT_PATH + c + "/")
            stat_df = pd.DataFrame(output, columns=['corpus', 'algo_name', 'total_intents', 'max intents found',
                                                    'minimum cluster count', 'average cluster count',
                                                    'avg nmi', 'max nmi', 'avg ars', 'max ars', 'avg acc', 'max acc',
                                                    'f1 score', 'max f1 score'])
            stat_df.to_excel(RESULT_PATH + c + "/stat.xlsx", index=False)



import copy
from functools import partial
import itertools
import math
from os import listdir, path
import random
import time
import dgl
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from tqdm import tqdm
from pytorchtools import EarlyStopping

from loader import GenericPapers2Graphs

from paths import CHECKPOINT, MATRIX, RESULTS, TRAINING_MODELS, DATA

classification = {'Observation': 0, 'Input': 1, 'Example': 2, 'Other': 3, '': 4}

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.readout_nodes(g, 'h', op='mean')
            return self.classify(hg)


class Model():
    def __init__(self, in_feats: int, hidden_layer_dim: int, n_classes: int, gpu: int) -> None:
        self.in_feats = in_feats
        self.hidden_layer_dim = hidden_layer_dim
        self.n_classes = n_classes

        self.gpu = gpu
    
    def init_model(self, is_new: bool):
        if is_new:
            print('─' * 40)
            print('Creating a model ({}, {}, {})'.format(self.in_feats, self.hidden_layer_dim, self.n_classes))
        if self.gpu >= 0 and torch.cuda.is_available():
            if is_new:
                print(' -> USING GPU\n')
            return Classifier(self.in_feats, self.hidden_layer_dim, self.n_classes).cuda(device=self.gpu)
        else:
            if is_new:
                print(' -> USING CPU\n')
            return Classifier(self.in_feats, self.hidden_layer_dim, self.n_classes).cpu()
            

class ModelPredict():
    def __init__(self, data: GenericPapers2Graphs, model: Model, training_save_dir = TRAINING_MODELS) -> None:
        self.data = data
        self.config = data.get_config()
        self.training_save_dir = training_save_dir

        self.model: Model = model

        self.train_graphs: list = []
        self.train_labels: list = []
        self.train_indices: list = []

        self.test_graphs: list = []
        self.test_labels: list = []
        self.test_indices: list = []
    
    
    def split_training_and_test(self, balance: bool = True, get_info: bool = True):
        if balance:
            train, test = self.data.balanced_split()
        else:
            train, test = self.data.random_split()
        
        if get_info:
            # print split value
            print("Train set:")
            self.data.split_info(train[0], train[1])
            print("Test set:")
            self.data.split_info(test[0], test[1])

        # get graphs and labels of train set
        self.train_graphs = train[0]
        self.train_labels = torch.tensor(train[1])
        self.train_indices = train[2]

        # get graphs and labels of test set
        self.test_graphs = test[0]
        self.test_labels = torch.tensor(test[1])
        self.test_indices = test[2]

    def k_fold(self, k: int, train_size: float = 0.9, seed: int = 0):
        X_train: list = []
        y_train: list = []
        idx_train: list = []

        X_val: list = []
        y_val: list = []
        idx_val: list = []

        kf = ShuffleSplit(n_splits=k, train_size=train_size, random_state=seed)

        for train_index, val_index in kf.split(self.train_graphs):
            X_train.append([self.train_graphs[i] for i in train_index])
            X_val.append([self.train_graphs[i] for i in val_index])

            y_train.append(torch.tensor([self.train_labels[i] for i in train_index]))
            y_val.append(torch.tensor([self.train_labels[i] for i in val_index]))

            idx_train.append(train_index.tolist())
            idx_val.append(val_index.tolist())

        print("Created {} fold (every fold has {} graphs for training and {} for validation).\n".format(str(k), str(len(X_train[0])), str(len(X_val[0]))))
        return (X_train, y_train, idx_train), (X_val, y_val, idx_val)


    def data_augmentation(self, dim_output: int, seed: int = 0, rc: bool = True):
        print("AUGMENTING DATA")
        augment_data: DataAugmentation = DataAugmentation(self.data)

        graphs: dict = {} # dictonary of all training data divided for annotated class
        for i, label in enumerate(self.train_labels):
            if label.item() not in graphs.keys():
                graphs[label.item()] = []
            graphs[label.item()].append({'graph': self.train_graphs[i], 'index': self.train_indices[i]})

        for i in graphs.keys():
            random.seed(seed)
            graph_samples: list = random.choices(graphs[i], k=int(dim_output/self.model.n_classes - len(graphs[i])))

            for sample in graph_samples:
                if rc:
                    if len(augment_data.detect_columns(sample['index'])) == 1:
                        new_graph = augment_data.invert_row(sample['index'], False)
                    elif len(augment_data.detect_rows(sample['index'])) == 1:
                        new_graph = augment_data.invert_column(sample['index'], False)
                    else:
                        # only invert rows and columns
                        functions_list: list = [
                            partial(augment_data.invert_column, sample['index'], False),
                            partial(augment_data.invert_row, sample['index'], False)
                        ]

                        new_graph = random.choice(functions_list)()

                else:
                    # all operations
                    functions_list: list = [
                            partial(augment_data.remove_node, sample['index'], random.randint(1, int(sample['graph'].num_nodes()*0.2)), False),
                            partial(augment_data.remove_edges, sample['index'], random.randint(1, int(sample['graph'].num_nodes()*0.2)), False),
                            partial(augment_data.invert_column, sample['index'], 2, False),
                            partial(augment_data.invert_row, sample['index'], 2, False)
                        ]

                    try:
                        new_graph = random.choice(functions_list)()
                    except:
                        new_graph = random.choice([functions_list[0], functions_list[1]])()

                self.train_graphs.append(new_graph)
                self.train_labels = torch.cat((self.train_labels, torch.tensor([self.data.labels[sample['index']]])))
                self.train_indices.append(sample['index'])        

            print("Adding {} for {} class.".format(str(int(dim_output/self.model.n_classes - len(graphs[i]))), next(j[0] for j in classification.items() if j[1] == i)))
        print("\n")
        

    def train_model(self, k: int, patience: int, batch_size: int, n_epochs: int, name_model: str, save: bool = True):
        # create model
        model: Classifier = self.model.init_model(True)

        # get k_fold subsample
        training, validation = self.k_fold(k)

        train_graphs: list = training[0]
        train_labels: list = training[1]

        val_graphs: list = validation[0]
        val_labels: list =  validation[1]

        # check batch size
        if not batch_size:
            batch_size = len(train_graphs[0]) # full batch size
        elif batch_size > len(train_graphs[0]):
            print("Batch size must be lower or equal of train graphs dimension (you insert {}, but train graphs dimension is {})".format(batch_size, len(train_graphs)))
            return

        print("TRAINING MODEL:")

        # parameters
        start_time = time.time()
        best_validation_loss: list = []
        best_validation_accuracy: list = []
        best_training_loss: list = []
        best_training_accuracy: list = []
        best_model: list = []

        for i in range(0, k):
            print(" -> {} folder (epochs: {}, batch size: {}, train set size: {}):".format((i+1), n_epochs, batch_size, len(train_graphs[i])))

            # initialize early stopper
            early_stopping = EarlyStopping(patience=patience, verbose=False, path=f"{CHECKPOINT}/checkpoint.pt")

            opt = torch.optim.Adam(model.parameters()) # specify optimize

            # calculate weights of classes
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels[i]), y=train_labels[i].tolist())

            for epoch in tqdm(range(n_epochs)):
                # randomize training
                randomized_train: list = random.sample(range(0, len(train_graphs[i])), len(train_graphs[i]))

                # get randomized training list
                random_train_graphs = [train_graphs[i][j] for j in randomized_train]
                random_train_labels = torch.tensor([train_labels[i][j] for j in randomized_train])

                model.train() # prep model for training
                for b in range(int(len(random_train_graphs) / batch_size)):
                    # get subsample of train_graphs       
                    train_batch = random_train_graphs[b * batch_size: min((b+1)*batch_size, len(random_train_graphs))]

                    # create batch graph from train batch graphs
                    batch_graphs = dgl.batch(train_batch)#.to(self.model.gpu)
                    batch_labels = random_train_labels[b * batch_size: min((b+1)*batch_size, len(random_train_labels))]#.to(self.model.gpu)

                    batch_features = batch_graphs.ndata['feat']#.to(self.model.gpu)

                    # calculate prediction
                    train_logits = model(batch_graphs, batch_features)
                    train_loss = F.cross_entropy(train_logits, batch_labels, weight=torch.tensor(class_weights, dtype=torch.float32)) 

                    # evaluate results
                    train_preds = train_logits.argmax(dim=1)
                    train_accuracy = torch.sum(train_preds == batch_labels).item() / len(batch_labels)

                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                # validate model
                val_loss, val_accuracy = self.val_model(model, val_graphs[i], val_labels[i])

                early_stopping(model, val_loss)
                if early_stopping.early_stop:
                    break
            
            # print model training results
            if epoch + 1 == n_epochs:
                print("Completed all {} epochs.".format(n_epochs))
            else:
                print("Validation loss not decrease for {} times. Break at {}/10000 epoch.".format(early_stopping.counter, epoch))                
            
            print("Results:\n-train loss: {}\n-train accuracy: {:.4f}\n-val loss: {}\n-val accuracy: {:.4f}".format(train_loss.item(), train_accuracy, val_loss.item(), val_accuracy))

            # append results
            best_validation_loss.append(val_loss)
            best_validation_accuracy.append(val_accuracy)
            best_training_loss.append(train_loss)
            best_training_accuracy.append(train_accuracy)
            best_model.append(model)     
            
            model = self.model.init_model(False) # reset model

        # get best result
        val, idx = min((val, idx) for (idx, val) in enumerate(best_validation_loss))
        val_loss = best_validation_loss[idx]
        val_accuracy = best_validation_accuracy[idx]
        train_loss = best_training_loss[idx]
        train_accuracy = best_training_accuracy[idx]
        model = best_model[idx]

        print("Time to training model: {:.4f}s.\nResults (best fold {}):\n-train loss: {:.5f}\n-train accuracy: {:.4f}\n-val loss: {:.5f}\n-val accuracy: {:.4f}\n".format(time.time() - start_time, idx + 1, train_loss.item(), train_accuracy, val_loss, val_accuracy))
        
        # save model
        if save:
            save_path = path.join(self.training_save_dir, name_model + '.pt')
            print("Saving training model ({})...\n". format(save_path))
            torch.save(model.state_dict(), save_path)
        
        return
        
    def val_model(self, model: Classifier, val_graphs: list, val_labels: list):
        batch_graphs = dgl.batch(val_graphs)

        model.eval() # prep model for evaluation
        with torch.no_grad():
            # get features
            batch_features = batch_graphs.ndata['feat']

            # calculate prediction
            val_logits = model(batch_graphs, batch_features)
            val_loss = F.cross_entropy(val_logits, val_labels)

            # evaluate results
            val_preds = val_logits.argmax(dim=1)
            val_accuracy = torch.sum(val_preds == val_labels).item() / len(val_labels)

        return val_loss, val_accuracy

    def test_model(self, name_model: str):
        model: Classifier = self.model.init_model(False)
        
        # load existing training model
        if name_model + '.pt' in listdir(self.training_save_dir):
            print("Loading training model ({})...\n".format(name_model))
            model.load_state_dict(torch.load(path.join(self.training_save_dir, name_model + '.pt')))
        else:
            print("model {}.pt not found. Training this model before.".format(name_model))
            return

        print("TESTING MODEL training{}.pt".format(name_model))

        start_time = time.time()
        batch_graphs = dgl.batch(self.test_graphs)

        model.eval()
        with torch.no_grad():
            # get features
            feats = batch_graphs.ndata['feat']

            # calculate prediction
            logits = model(batch_graphs, feats)

            # evaluate results
            pred = logits.argmax(dim=1)

            test_accuracy = torch.sum(pred == self.test_labels).item() / len(self.test_labels)

            results = precision_recall_fscore_support(self.test_labels, pred)
            print("Results")
            print(results)
            print("-Precision:", results[0])
            print("-Recall:", results[1])
            print("-F1 Score:", results[2])
            print("-Support:", results[3])
            
            results_micro = precision_recall_fscore_support(self.test_labels, pred, average="micro")
            print("Micro")
            print(results_micro)
            print("-Precision:", results_micro[0])
            print("-Recall:", results_micro[1])
            print("-F1 Score:", results_micro[2])

            results_macro = precision_recall_fscore_support(self.test_labels, pred, average="macro")
            print("Macro")
            print(results_macro)
            print("-Precision:", results_macro[0])
            print("-Recall:", results_macro[1])
            print("-F1 Score:", results_macro[2])

            print("\n")
            print(classification_report(self.test_labels, pred, target_names=['Observation', 'Input', 'Example', 'Other']))

            cm = confusion_matrix(self.test_labels, pred, normalize='true')
            cmd = ConfusionMatrixDisplay(cm, display_labels=['Observation', 'Input', 'Example', 'Other'])
            cmd.plot(cmap=plt.cm.Blues)
            plt.savefig(f"{MATRIX}/{name_model}", bbox_inches = "tight")

        # print model testing results
        print("Time to testing model: {:.4f}s.\nResults:\n-correct predicted: {}\{}\n-accuracy: {:.4f}\n".format(time.time() - start_time, int(test_accuracy*len(self.test_labels)), len(self.test_labels), test_accuracy))


class DataAugmentation():
    def __init__(self, data: GenericPapers2Graphs) -> None:
        self.data = data


    def __swap(self, graph, idx_node1: int, idx_node2: int, verbose: bool = False):
        tmp = copy.deepcopy(graph.ndata['feat'][idx_node1][13:])
        graph.ndata['feat'][idx_node1][13:] = graph.ndata['feat'][idx_node2][13:]
        graph.ndata['feat'][idx_node2][13:] = tmp
        if verbose:
            print('swapped node {} with node {}'.format(idx_node1, idx_node2))

        return graph

    def detect_columns(self, idx: int) -> list:
        page = self.data.pages[idx]

        width_page = [0] * 596

        bboxs = page['bboxs']
        for box in bboxs:
            x1 = box[0]
            x2 = box[2]

            for i in range(x1, x2):
                width_page[i] = 1

        column_coordinates: list = []
        for i in range(0, len(width_page)-1):
            if width_page[i]==0 and width_page[i+1]==1:
                c: list = [i]
            elif width_page[i]==1 and width_page[i+1]==0:
                c.append(i)
                column_coordinates.append(c)
        return column_coordinates # return indices of boxs separate from columns

    def detect_rows(self, idx: int) -> list:
        import json
        annotation = json.load(open(f"{DATA}/metadata.json", 'r'))

        page = self.data.pages[idx]

        page_id = page['page'].split('-Page')[0]
        pagination = int(page['page'].split('-Page')[1].split('-Table')[0])
        num_table = page['page'].split('-Table')[1].replace('.pdf', '')

        table = next(t for t in annotation[page_id] if (t['Table'] == num_table and t['pagination'] == pagination))
        header_row = table['headerRowCount']

        rows_coordinates: list = []
        for i, box in enumerate(page['bboxs'][:-1]):
            y1 = box[1]
            y2 = box[3]
            if i == 0 or box[0] < page['bboxs'][i - 1][0]:
                r: list = [y1]
            elif box[0] > page['bboxs'][i + 1][0] or i+1==len(page['bboxs']) - 1:
                r.append(y2)
                rows_coordinates.append(r)

        return rows_coordinates[header_row:] # return indices of boxs separate from rows without header row(s)
   

    def invert_column(self, idx: int, verbose: bool = False, seed: int = 0):
        def inside(column_x1: int, column_x2: int, box_x1: int, box_x2: int) -> bool:
            center_box = box_x2 - (box_x2 - box_x1)/2

            if (column_x1 <= center_box <= column_x2):
                return True
            return False
 
        detected_columns: list = self.detect_columns(idx) # detect columns structure of a table

        if(len(detected_columns) == 1):
            raise Exception("Cannot swap single column")
        
        columns = [[] for i in range(0, len(detected_columns))] # divide columns depending on detected structure
        for i, box in enumerate(self.data.pages[idx]['bboxs']):
            for j, column in enumerate(detected_columns):
                if inside(column[0], column[1], box[0], box[2]):
                    columns[j].append(i)
        
        random.seed(seed)
        n_swap: int =  random.randint(1, math.comb(len(detected_columns), 2)) # random number of swap (from 1 to simple combination without repetition of number columns took 2 columns at times)
        random_swap = random.sample(list(itertools.combinations(range(len(detected_columns)), 2)), n_swap) # random sample of columns to swap

        graph = copy.deepcopy(self.data.graphs[idx])
        for cols_to_swap in random_swap:
            if verbose:
                print("swap column {} with {} from graph {}".format(cols_to_swap[0], cols_to_swap[1], idx))
            for i, j in zip(columns[cols_to_swap[0]], columns[cols_to_swap[1]]):
                graph = self.__swap(graph, i, j)

        return graph
        
    def invert_row(self, idx: int, verbose: bool = False, seed: int = 0):
        def inside(row_y1: int, row_y2: int, box_y1: int, box_y2: int) -> bool:
            center_box = box_y2 - (box_y2 - box_y1)/2

            if (row_y1 <= center_box <= row_y2):
                return True
            return False
        
        detected_rows: list = self.detect_rows(idx) # detect rows structure of a table

        if(len(detected_rows) == 1):
            raise Exception("Cannot swap single row")
        
        rows = [[] for i in range(0, len(detected_rows))] # divide rows depending on detected structure
        for i, box in enumerate(self.data.pages[idx]['bboxs']):
            for j, row in enumerate(detected_rows):
                if inside(row[0], row[1], box[1], box[3]):
                    rows[j].append(i)

        random.seed(seed)
        n_swap: int =  random.randint(1, math.comb(len(detected_rows), 2)) # random number of swap (from 1 to simple combination without repetition of number rows took 2 rows at times)
        random_swap = random.sample(list(itertools.combinations(range(len(detected_rows)), 2)), n_swap) # random sample of rows to swap
        
        
        graph = copy.deepcopy(self.data.graphs[idx])
        for rows_to_swap in random_swap:
            if verbose:
                print("swap row {} with {} from graph {}".format(rows_to_swap[0], rows_to_swap[1], idx))
            for i, j in zip(rows[rows_to_swap[0]], rows[rows_to_swap[1]]):
                graph = self.__swap(graph, i, j)

        return graph


    def remove_node(self, idx: int, dim_node_to_remove: int, verbose: bool = False, seed: int = 0):
        nodes = self.data.graphs[idx].num_nodes()

        random.seed(seed)
        nodes_to_remove = random.sample(range(0, nodes), dim_node_to_remove)
        if verbose:
            print("Deleted {} nodes from graph {}".format(dim_node_to_remove, idx))

        return dgl.remove_nodes(self.data.graphs[idx], torch.tensor(nodes_to_remove, dtype=torch.int32))
    
    def remove_edges(self, idx: int, dim_edges_to_remove: int, verbose: bool = False, seed: int = 0):
        graph = dgl.remove_self_loop(self.data.graphs[idx])
        edges = graph.num_edges()

        random.seed(seed)
        edges_to_remove = random.sample(range(0, edges), dim_edges_to_remove)
        if verbose:
            print("Deleted {} edges from graph {}".format(dim_edges_to_remove, idx))

        
        graph = dgl.remove_edges(graph, torch.tensor(edges_to_remove, dtype=torch.int32))
        return dgl.add_self_loop(graph)


if __name__=='__main__':
    results_save_dir = RESULTS
    model_save_dir = TRAINING_MODELS

    import sys
    results_number = len([i for i, x in enumerate(listdir(results_save_dir)) if 'results' in x])
    sys.stdout = open(f'{RESULTS}/results{str(results_number)}.txt', 'w', encoding="utf-8")


    # loading data
    data = GenericPapers2Graphs()

    # loading modifying data
    data.modify_graphs()
    
    # delete uncorrected annotated graphs
    data.delete_not_correct_graphs(4)

    # model hyperparameters
    gpu = -1
    in_feats = 313 # change to 213 for sciSpacy
    hidden_layer_dim = 150
    n_classes = 4
    k_fold = 2
    patience = 10
    n_epochs = 10000    
    

    # create model
    model = Model(in_feats, hidden_layer_dim, n_classes, gpu=gpu)
    model_predict = ModelPredict(data, model)
    model_predict.split_training_and_test(balance=True, get_info=True)

    # augment data
    model_predict.data_augmentation(200, False)
    

    # model full batches
    model_number = len([i for i, x in enumerate(listdir(model_save_dir)) if 'fullbatch_' in x])
    name_model = 'model_fullbatch_' + str(model_number)

    model_predict.train_model(k=k_fold, patience=patience, batch_size=None, n_epochs=n_epochs, name_model=name_model, save=True)
    model_predict.test_model(name_model=name_model)

    
    # model mini batches (8)
    batch_size = 8
    model_number = len([i for i, x in enumerate(listdir(model_save_dir)) if 'sizebatch' + str(batch_size) + '_' in x])
    name_model = 'model_sizebatch' + str(batch_size) + '_' + str(model_number)

    model_predict.train_model(k=k_fold, patience=patience, batch_size=batch_size, n_epochs=n_epochs, name_model=name_model, save=True)
    model_predict.test_model(name_model=name_model)
    
    '''
    # model mini batches (30)
    batch_size = 30
    model_number = len([i for i, x in enumerate(listdir(model_save_dir)) if 'sizebatch' + str(batch_size) + '_' in x])
    name_model = 'model_sizebatch' + str(batch_size) + '_' + str(model_number)

    model_predict.train_model(k=k_fold, patience=patience, batch_size=batch_size, n_epochs=n_epochs, name_model=name_model, save=True)
    model_predict.test_model(name_model=name_model)


    # model single batches
    batch_size = 1
    model_number = len([i for i, x in enumerate(listdir(model_save_dir)) if 'sizebatch' + str(batch_size) + '_' in x])
    name_model = 'model_sizebatch' + str(batch_size) + '_' + str(model_number)

    model_predict.train_model(k=k_fold, patience=patience, batch_size=batch_size, n_epochs=n_epochs, name_model=name_model, save=True)
    model_predict.test_model(name_model=name_model)
    '''
    sys.stdout.close()

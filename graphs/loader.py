import random
import time
import sys
import dgl
import os
import json
from dgl.data import DGLDataset
from pdf2image.pdf2image import convert_from_path
import torch
from tqdm import tqdm
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from sklearn.model_selection import train_test_split


import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from paths import DATA, CONFIG, IMGS, RAW, GRAPHS
from graphs.builder import GraphBuilder

classification = {'Observation': 0, 'Input': 1, 'Example': 2, 'Other': 3, '': 4}

class GenericPapers2Graphs(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    """
    def __init__(self,
                 num_papers = -1,
                 config = CONFIG / "graph.yaml",
                 raw_dir = RAW,
                 save_dir = GRAPHS,
                 out_dir = IMGS):
    
        self.out_dir = out_dir
        self.num_papers = num_papers
        self.gb = GraphBuilder(config=config)
        self.graphs = []
        self.labels = []
        self.pages = []
        self.seed = self.gb.get_config().PREPROCESS.seed        
        random.seed(self.seed)
        super(GenericPapers2Graphs, self).__init__(name = '', raw_dir = raw_dir, save_dir = save_dir, verbose = False)
           
    def __getitem__(self, idx):
        # get single graph (in idx position) with his label
        return self.graphs[idx],  torch.tensor(self.labels[idx])

    def __len__(self):
        # get number of graphs
        return len(self.graphs)

    def get_config(self):
        # get config file
        return self.gb.get_config()
    
    def get_gb(self):
        # get graph builder
        return self.gb

    def get_info(self, g, num_graph: int):
        # print processed data in imgs folder (create img of processed file qith graph of table)
        graph = dgl.to_simple(g)
        page = self.pages[num_graph]['page']
        bboxs = self.pages[num_graph]['bboxs']

        page_path = os.path.join(self.raw_dir, page)
        out_path = os.path.join(self.out_dir, page)

        print(page_path, out_path)
        
        self.gb.print_graph(graph, None, bboxs, page_path=page_path, out_path=out_path, node_labels=False)


    def save(self):
        # save processed data to directory `self.save_path`
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        print(graph_path, info_path)
        save_graphs(graph_path, self.graphs, {'labels': torch.tensor(self.labels)})
        save_info(info_path, {'pages': self.pages})

    def load(self):
        # load processed data from directory `self.save_path`
        print(f"\nLoading processed data ... ", end='')
        start_time = time.time()
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels'].tolist()
        self.pages = load_info(info_path)['pages']
        print("took %ss" % (round(time.time() - start_time, 2)))

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)


    def process(self):
        # processing raw data from raw_dir, and saving graph/s to save_dir
        for page in tqdm(os.listdir(self.raw_dir)):
            page_path = os.path.join(self.raw_dir, page)
            g, bboxs, texts = self.gb.get_graph(page_path, None, self.get_config().PREPROCESS.mode, set_labels=False)
            if g is not None:
                self.graphs.append(g)
                self.labels.append(5) # set 5 as default class None (initial classification for table is None)
                self.pages.append({'page': page, 'bboxs' : bboxs, 'texts' : texts})

    def modify_graphs(self, load_features = False, save=True):
        #check if center of bbox is inside annotation bbox (of table)
        def inside_box(annotation_bbox: list, bbox: list) -> bool:
            center_bbox = [bbox[2] - (bbox[2] - bbox[0])/2, bbox[3]  - (bbox[3] - bbox[1])/2]

            return annotation_bbox[0] <= center_bbox[0] <= annotation_bbox[2] and annotation_bbox[1] <= center_bbox[1] <= annotation_bbox[3]

        print(f"\nModifying graphs ... ", end='\n')
        start_time = time.time()

        annotation_path = f"{DATA}/metadata.json"
        annotation = json.load(open(annotation_path, 'r'))

        graph_path = f"{self.save_path}graphs_modified.bin"
        info_path = f"{self.save_path}info_modified.pkl"

        #load graph and info if already processed
        if os.path.exists(graph_path) and os.path.exists(info_path):
            print("\n -> Loading already modified graphs ... ", end='')
            start_time = time.time()
            self.graphs, label_dict = load_graphs(graph_path)
            self.labels = label_dict['labels'].tolist()
            self.pages = load_info(info_path)['pages']
            print("took %ss" % (round(time.time() - start_time, 2)))
            return

        for num_graph in range(0, self.__len__()):
            print(" -> Progress {}/{} | {}".format(num_graph+1, self.__len__(), round(time.time() - start_time, 2)), end='\r')

            execute_time = time.time()

            # lists
            bboxs = self.pages[num_graph]['bboxs']
            texts = self.pages[num_graph]['texts']

            # set node and edge features
            page = self.pages[num_graph]['page']
            page_id = page.split('-Page')[0]
            pagination = int(page.split('-Page')[1].split('-Table')[0])
            num_table = page.split('-Table')[1].replace('.pdf', '')

            page_path = os.path.join(self.raw_dir, page)
            size = convert_from_path(page_path)[0].size
            self.gb.set_features(self.graphs[num_graph], bboxs, texts, size, embedder='sci')
            
            table = next(t for t in annotation[page_id] if (t['Table'] == num_table and t['pagination'] == pagination))

            nodes = self.graphs[num_graph].num_nodes()
                
            nodes_to_remove = []
            bboxs_update, texts_update = bboxs, texts
            annotation_bbox = [table['regionBoundary']['x1'], table['regionBoundary']['y1'], table['regionBoundary']['x2'], table['regionBoundary']['y2']]
                
            for node in range(0, nodes):
                if not inside_box(annotation_bbox, bboxs[node]):
                    nodes_to_remove.append(node)
                    bboxs_update[node] = None
                    texts_update[node] = None
                
            bboxs = [bbox for bbox in bboxs_update if bbox != None]
            texts = [text for text in texts_update if text != None]
            
            self.pages[num_graph] = {'page': page, 'bboxs' : bboxs, 'texts' : texts}
            self.graphs[num_graph] = dgl.remove_nodes(self.graphs[num_graph], torch.tensor(nodes_to_remove, dtype=torch.int32))

            self.graphs[num_graph] = dgl.add_self_loop(self.graphs[num_graph])

            self.labels[num_graph] = classification[table['type']]
                    
            print()
            print("took %ss" % (round(time.time() - execute_time, 2)))
 
            # print graph
            self.get_info(self.graphs[num_graph], num_graph)
            
        if save:
            print("Saving modified graphs.\n")
            save_graphs(graph_path, self.graphs, {'labels': torch.tensor(self.labels)})
            save_info(info_path, {'pages': self.pages})

        return
    
    def delete_not_correct_graphs(self, not_correct_annotation: int):
        dimension = self.__len__()

        # delete not correct annotated graphs and empty graphs and relative information
        graphs: list = []
        pages: list = []
        labels: list = []
        for i in range(dimension):
            if ((not self.labels[i] == not_correct_annotation) and (not self.graphs[i].num_nodes() == 0)):
                graphs.append(self.graphs[i])
                pages.append(self.pages[i])
                labels.append(self.labels[i])
        
        self.graphs = graphs
        self.labels = labels
        self.pages = pages

        #delete not correct annotation from calssification dict
        del classification[list(classification.keys())[list(classification.values()).index(not_correct_annotation)]]

        print("\nDeleted {} not correct graphs. New graphs dimension is {} (before was {}).\n".format(dimension - self.__len__(), self.__len__(), dimension))
        return

    
    def random_split(self): 
        # generate random split of dataset in training and testing set
        rate = self.get_config().DLTRAIN.rate
        
        num_graphs = self.__len__()
                
        train_amount = int(num_graphs * rate)
        test_amount = num_graphs - train_amount
        print(" -> Split {} pages : Train {} | Val {}.\n".format(num_graphs, train_amount, test_amount))
            
        train_indices = random.sample(range(0, num_graphs), train_amount)
        test_indices = list(set(range(0, num_graphs)) - set(train_indices))
            
        train_graphs = []
        train_labels = []
        test_graphs = []
        test_labels = []
            
        for i in train_indices:
            train_graphs.append(self.graphs[i])
            train_labels.append(self.labels[i])
                
        for i in test_indices:
            test_graphs.append(self.graphs[i])
            test_labels.append(self.labels[i])
                    
        return (train_graphs, train_labels, train_indices), (test_graphs, test_labels, test_indices)
        
    def balanced_split(self):
        # generate balanced split of dataset in training and testing set
        rate = self.get_config().DLTRAIN.rate
        num_graphs: int = self.__len__()
        indices: list = range(len(self.graphs))

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(self.graphs, self.labels, indices, train_size=rate, random_state=0, stratify=self.labels)

        print(" -> Split {} pages : Train {} | Test {}.\n".format(num_graphs, len(X_train), len(X_test)))

        return (X_train, y_train, idx_train), (X_test, y_test, idx_test)

    def split_info(self, split_graphs, split_labels):
        l = {}
        for c in classification:
            l[c] = [g for i, g in enumerate(split_graphs) if split_labels[i] == classification[c]]
            print("{}: {}/{} ({}%)".format(c, len(l[c]), len(split_graphs), round((len(l[c])/len(split_graphs))*100, 0)))
        print("")
        return l

if __name__ == "__main__":
    start_time = time.time()
    data = GenericPapers2Graphs()
    print("%ss" % (round(time.time() - start_time, 2)))
    data.modify_graphs()
    
    

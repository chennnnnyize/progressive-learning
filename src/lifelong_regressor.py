from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.random import set_seed
from sklearn.base import clone 
from sklearn.neighbors import KNeighborsRegressor

class LifeLongRegression():
    def __init__(self, acorn = None, verbose = False):
        self.X_across_tasks = {}
        self.y_across_tasks = {}
        
        self.X_across_tasks_scaled = {}
        self.y_across_tasks_scaled = {}
        
        self.models = {}  
        self.task_voters = defaultdict(list)
        self.task_voters_knn = defaultdict(list)
        #for completeness, need to train task decider for every combination
        #of representations, but here only doing it for all
        self.task_deciders = {}
        self.task_deciders_knn = {}
        self.n_tasks = 0
                
        if acorn is not None:
            np.random.seed(acorn)
            set_seed(acorn)
            
        self.verbose = verbose
        
    def check_task_idx_(self, task_idx):
        if task_idx >= self.n_tasks:
            raise Exception("Invalid Task IDX")
    
    def new_task(self, 
                   X, 
                   y, 
                   epochs = 100, 
                   lr = 5e-4, 
                   max_samples = .63,
                   encoder_dim = 4,
                   auto_encoder_transformer=True):
        
        
        self.X_across_tasks[self.n_tasks] = X
        self.y_across_tasks[self.n_tasks] = y

        model = HonestRegression(verbose=self.verbose, calibration_split=max_samples, encoder_dim=encoder_dim, 
                                 auto_encoder_transformer=auto_encoder_transformer)
        model.fit(X, y)
        self.models[self.n_tasks] = model 
        knn_voter = model.knn
        
        #train voters using current task's transformers for all previous tasks
        for task in range(self.n_tasks):
            X_task_trans = model.transform(self.X_across_tasks[task])
            y_task = model.scale_y(self.y_across_tasks[task])
            
            task_voter = NNVoter(X_task_trans, y_task,  
                validation_split = 1-max_samples,
                input_dim = model.encoder_dim,
                verbose = model.verbose,
                epochs = epochs,
                lr = lr)
            
            task_knn_voter = clone(knn_voter)
            task_knn_voter.fit(X_task_trans, y_task)
        
            self.task_voters_knn[task].append((self.n_tasks, task_knn_voter))
            self.task_voters[task].append((self.n_tasks, task_voter))
        
        #train voters for current task using prevoious tasks' transformer
        for task in range(self.n_tasks+1):
            task_model = self.models[task]
            task_knn_voter = clone(task_model.knn)
            
            X_trans = task_model.transform(X)
            y_trans = task_model.scale_y(y)
            
            task_voter =  NNVoter(X_trans, y_trans,  
                validation_split = 1-max_samples,
                input_dim = task_model.encoder_dim,
                verbose = task_model.verbose,
                epochs = epochs,
                lr = lr)
            
            task_knn_voter.fit(X_trans, y_trans)
            self.task_voters_knn[self.n_tasks].append((task, task_knn_voter))
            self.task_voters[self.n_tasks].append((task, task_voter)) 
        
        #train decidersfor 
        self._train_task_deciders(voter_name='knn')
        self._train_task_deciders(voter_name='mlp')
        
        self.n_tasks += 1
    
    #TODO:: for completeness we need train a decider for each, combinatorion of representations
    #but here just doing it for "all"
    def _train_task_deciders(self, voter_name='knn'):
        if voter_name == 'knn':
            task_voters = self.task_voters_knn
        else:
            task_voters = self.task_voters
            
        for task, voters in task_voters.items():
            X = self.X_across_tasks[task]
            y = self.y_across_tasks[task]
            votes = []
            for representation, voter in voters:
                model = self.models[representation]
                X_trans = model.transform(X)
                vote = voter.predict(X_trans)
                votes.append(model.inverse_scale_y(vote))
            
            n_neighbors = max(16 * int(np.log2(len(votes))), 1)
            decider = KNeighborsRegressor(n_neighbors, weights = "distance", p = 1) 
            votes = np.hstack(votes)
            decider.fit(votes, y)
            
            if voter_name =='knn':
                self.task_deciders_knn[task] = decider
            else:
                self.task_deciders[task] = decider
                
                
                
    def _estimate_posteriors(self, X, representation = 0, decider = 0, use_knn=False):
        
        self.check_task_idx_(decider)
        use_combine_decider = False
    
        #!!!!this function will break when representation is not "all" or a singleton 
        if representation == "all":
            wanted_representation = set(range(self.n_tasks))
            use_combine_decider = True
        elif isinstance(representation, int):
            wated_representation = set([representation])
        
        
        if use_knn:
            voters = self.task_voters_knn[decider]
            combine_decider = self.task_deciders_knn[decider]
        else:
            voters = self.task_voters[decider]
            combine_decider = self.task_deciders[decider]
        
        
        votes = []
        accuracy_total = 0

        for representation, voter in voters:
            #!!!For the combine_decider to work, must use all know voters for a task
#             if not representation in wanted_representation:
#                 continue            
            model = self.models[representation]
            X_trans = model.transform(X)
            vote = voter.predict(X_trans)
            votes.append(model.inverse_scale_y(vote)) 
        
        if not use_combine_decider:
            return votes[0]
        else:
            return combine_decider.predict(np.hstack(votes))
            
    def predict(self, X, representation = 0, decider = 0, use_knn=False):
        return self._estimate_posteriors(X, representation, decider, use_knn=use_knn)

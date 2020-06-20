import pandas as pd
from lifelong_regressor import LifeLongRegression

def load_data(file_name, train_size):
    with open(file_name, 'r') as handle:
        tmp = pd.read_csv(handle, header = None)
        target = tmp.loc[:, 0]
        features = tmp.loc[:,1:]
        return features[:train_size].to_numpy(), target[:train_size].to_numpy(), features[train_size:].to_numpy(), target[train_size:].to_numpy()

#use 60 days for training, which reminds 12 days for testing
train_size = 60 * 24

chicago_train_features, chicago_train_target, chicago_test_features, chicago_test_target = \
    load_data('/mnt/c/Users/weiwya/Desktop/UW_c19/Data_and_Simulations/Data_Processed/Chicago_mobility_all.csv', train_size)

nyc_train_features, nyc_train_target, nyc_test_features, nyc_test_target = \
    load_data('/mnt/c/Users/weiwya/Desktop/UW_c19/Data_and_Simulations/Data_Processed/NY_mobility_all.csv', train_size)

seattle_train_features, seattle_train_target, seattle_test_features, seattle_test_target = \
    load_data('/mnt/c/Users/weiwya/Desktop/UW_c19/Data_and_Simulations/Data_Processed/Seattle_mobility_all.csv', train_size)

phil_train_features, phil_train_target, phil_test_features, phil_test_target = \
    load_data('/mnt/c/Users/weiwya/Desktop/UW_c19/Data_and_Simulations/Data_Processed/Phil_mobility_all.csv', train_size)


#add each city as a new task
learner = LifeLongRegression(acorn=1234)
learner.new_task(chicago_train_features, chicago_train_target, auto_encoder_transformer=False, encoder_dim=8, epochs=500)
print('done chicago')
ch_0 = learner.predict(chicago_test_features, representation=0, decider=0)
ch_1 = learner.predict(chicago_test_features, representation='all', decider=0)
print('chicago e0:%s e1:%s' %(calculate_MAPE(ch_0, chicago_test_target), calculate_MAPE(ch_1, chicago_test_target) )  )

learner.new_task(nyc_train_features, nyc_train_target, auto_encoder_transformer= False, encoder_dim=8, epochs=500)
print('done nyc')
ch_0 = learner.predict(chicago_test_features, representation=0, decider=0)
ch_1 = learner.predict(chicago_test_features, representation='all', decider=0)
print('chicago e0:%s e1:%s' %(calculate_MAPE(ch_0, chicago_test_target), calculate_MAPE(ch_1, chicago_test_target) )  )

learner.new_task(phil_train_features, phil_train_target, auto_encoder_transformer=False, encoder_dim=8, epochs=500)
print('done phil')
ch_0 = learner.predict(chicago_test_features, representation=0, decider=0)
ch_1 = learner.predict(chicago_test_features, representation='all', decider=0)
print('chicago e0:%s e1:%s' %(calculate_MAPE(ch_0, chicago_test_target), calculate_MAPE(ch_1, chicago_test_target) )  )

learner.new_task(seattle_train_features, seattle_train_target, auto_encoder_transformer=False, encoder_dim=8, epochs=500)
print('done seattle')
ch_0 = learner.predict(chicago_test_features, representation=0, decider=0)
ch_1 = learner.predict(chicago_test_features, representation='all', decider=0)
print('chicago e0:%s e1:%s' %(calculate_MAPE(ch_0, chicago_test_target), calculate_MAPE(ch_1, chicago_test_target) )  )


#check overall performance for all cities using all representation vs single representation

ch_0 = learner.predict(chicago_test_features, representation=0, decider=0, use_knn=False)
ch_1 = learner.predict(chicago_test_features, representation='all', decider=0, use_knn=False)
print('chicago e0:%s e1:%s' %(calculate_MAPE(ch_0, chicago_test_target), calculate_MAPE(ch_1, chicago_test_target) )  )


ny_0 = learner.predict(nyc_test_features, representation=1, decider=1, use_knn=False)
ny_1 = learner.predict(nyc_test_features, representation='all', decider=1, use_knn=False)
print('NYC e0:%s e1:%s' %(calculate_MAPE(ny_0, nyc_test_target), calculate_MAPE(ny_1, nyc_test_target) )  )


phi_0 = learner.predict(phil_test_features, representation=2, decider=2, use_knn=False)
phi_1 = learner.predict(phil_test_features, representation='all', decider=2, use_knn=False)
print('phil e0:%s e1:%s' %(calculate_MAPE(phi_0, phil_test_target), calculate_MAPE(phi_1, phil_test_target) )  )

seattle_0 = learner.predict(seattle_test_features, representation=3, decider=3, use_knn=False)
seattle_1 = learner.predict(seattle_test_features, representation='all', decider=3, use_knn=False)
print('Seattle e0:%s e1:%s' %(calculate_MAPE(seattle_0, seattle_test_target), calculate_MAPE(seattle_1, seattle_test_target) )  )

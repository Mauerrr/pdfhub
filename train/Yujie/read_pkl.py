import pickle
import sys
sys.path.append('/media/yujie/data/E2EAD/endtoenddriving/')
path_train = "/media/yujie/data/E2EAD/endtoenddriving/exp/submission_transformer8_greedy_decoder_ego_state_warmup/2024.05.04.20.52.36/submission.pkl"

f = open(path_train, 'rb')
data_train = pickle.load(f)
print(data_train)
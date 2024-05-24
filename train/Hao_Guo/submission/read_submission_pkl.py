import pickle

# 提交的pkl文件是只有trajectory的
data = pickle.load(open('/home/eze2szh/endtoenddriving/exp/submission_formal524ghgru/2024.05.24.15.49.37/submission.pkl', 'rb'))
print(data)

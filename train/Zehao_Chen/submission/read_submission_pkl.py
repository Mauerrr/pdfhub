import pickle

# 提交的pkl文件是只有trajectory的
data = pickle.load(open('/home/hguo/e2eAD/endtoenddriving/HuggingFace_Submission/HaoGuoFormalTrack/submission.pkl', 'rb'))
print(data)

import pickle
import json

data = pickle.load(open('C:/Users/heeryung/code/24w-Tri-Modalities/data/msrvtt_jsfusion_test.pkl', 'rb'))
anno = json.load(open('C:/Users/heeryung/code/24w-Tri-Modalities/data/test_videodatainfo.json', 'rb'))

new_data = []

for f in data:
    for a in anno['videos']:
        if f['id'][5:] == str(a['id']):
            f['category'] = a['category']
            new_data.append(f)

output_file = 'C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl'

with open(output_file, 'wb') as f:
    pickle.dump(new_data, f)
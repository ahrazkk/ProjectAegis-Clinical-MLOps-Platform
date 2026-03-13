import json
m = json.load(open(r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\public\models\pill-classifier\model.json'))
wm = m.get('weightsManifest', [])
print('Format:', m.get('format', 'unknown'))
for g in wm:
    paths = g.get('paths', [])
    print(f'  Shard paths: {paths}')
labels = json.load(open(r'c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\public\models\pill-classifier\labels.json'))
print(f'\nLabels: {len(labels)} classes')
print(f'Sample: {labels[:5]}...{labels[-3:]}')

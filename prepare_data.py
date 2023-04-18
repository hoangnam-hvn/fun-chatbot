import yaml
import re

with open('chatdata.yaml', 'r', encoding='utf-8') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)['Content']

questions = []

for i, qaa in enumerate(data):
    clean_question = re.sub(r'[^\w\s]', '', qaa[0].lower())
    questions.append(clean_question)
    
labels = list(range(len(questions)))
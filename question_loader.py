import pandas as pd
import ast

FEW_SHOT = """
######################
-Examples-
######################
Question -

A 50-year-old man comes to the physician because of a 6-month history of difficulties having sexual intercourse due to erectile dysfunction. He has type 2 diabetes mellitus that is well controlled with metformin. He does not smoke. He drinks 5–6 beers daily. His vital signs are within normal limits. Physical examination shows bilateral pedal edema, decreased testicular volume, and increased breast tissue. The spleen is palpable 2 cm below the left costal margin. Abdominal ultrasound shows an atrophic, hyperechoic, nodular liver. An upper endoscopy is performed and shows dilated submucosal veins 2 mm in diameter with red spots on their surface in the distal esophagus. Therapy with a sildenafil is initiated for his erectile dysfunction. Which of the following is the most appropriate next step in management of this patient's esophageal findings?

Choices -
A Injection sclerotherapy
B Nadolol therapy
C Losaratan therapy
D Octreotide therapy
E Isosorbide mononitrate therapy
F Endoscopic band ligation
G Transjugular intrahepatic portosystemic shunt
H Metoprolol therapy

As an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is
######################
B
######################
Question -

A 7-year-old boy is brought to the physician by his mother because his teachers have noticed him staring blankly on multiple occasions over the past month. These episodes last for several seconds and occasionally his eyelids flutter. He was born at term and has no history of serious illness. He has met all his developmental milestones. He appears healthy. Neurologic examination shows no focal findings. Hyperventilation for 30 seconds precipitates an episode of unresponsiveness and eyelid fluttering that lasts for 7 seconds. He regains consciousness immediately afterward. An electroencephalogram shows 3-Hz spikes and waves. Which of the following is the most appropriate pharmacotherapy for this patient?

Choices -

A Vigabatrin
B Lamotrigine
C Pregabalin
D Clonazepam
E Carbamazepine
F Ethosuximide
G Phenytoin
H Levetiracetam

As an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is
######################
F
######################
-Real Data-
######################
"""

# Load the data
data = pd.read_json(path_or_buf='./data_clean/questions/US/US_qbank.jsonl', lines=True)
print(data.head())

# Function to get the length of the data
def get_data_len():
    return len(data)

def get_is_correct_query(i):
    num_options = len(get_row_options(i))

    all_queries = []
    for _ in range(num_options):
        query = ''
        query += 'Question -\n\n' + get_row_question(i) + '\n\n'
        query += 'As an extremely experienced and knowledgeable medical professional answering this question accurately, is the correct answer ' + get_ith_option(i, _)  + '?'

        query += "Please answer with Yes or No."
        all_queries.append(query)

    return all_queries

# Function to get the query of a specific row
def get_row_query(i):
    query = ''
    # query += FEW_SHOT
    query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(get_row_options(i))
    query += '\nAs an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is '
    return query

# Function to get the question of a specific row
def get_row_question(i):
    return data['question'][i]

# Function to get the options of a specific row
def get_row_options(i):
    return data['options'][i]

# Function to get the correct answer of a specific row
def get_correct_answer(row):
    return data['answer'][row]

# Function to convert options to a formatted string
def getOptionsString(options):
    if isinstance(options, str):
        options = strToDict(options)
        
    res = ''
    for i in options:  
        res += i + ' ' + options[i].split('\n')[0]
        res += '\n'

    return res

def get_ith_option(row, i):
    options = get_row_options(row)
    if isinstance(options, str):
        options = strToDict(options)
        
    return options[i].split('\n')[0]


# Function to convert a string to a dictionary
def strToDict(Str):
    try:
        return ast.literal_eval(Str)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dictionary: {e}")
        return {}

if __name__ == '__main__':
    print(get_row_query(97))
    print(get_correct_answer(97))

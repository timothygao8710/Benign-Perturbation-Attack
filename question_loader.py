import pandas as pd
import ast
import random
import string

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]

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

random_state = 42

# Function to get the length of the data
def get_data_len():
    return len(data)

def get_is_correct_query(i):
    num_options = len(get_row_options(i))

    all_queries = []
    for option in range(num_options):
        query = ''
        query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(get_row_options(i))
        query += 'As an extremely experienced and knowledgeable medical professional answering this question accurately, is the correct answer ' + get_ith_option(i, option)  + '? '

        # query += "Please answer with Yes or No."
        all_queries.append(query)

    return all_queries

# Function to get the query of a specific row
def get_row_query(i):
    query = ''
    # query += FEW_SHOT
    query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(get_row_options(i), False)
    query += '\nAs an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is '
    return query

def get_shuffled_row_query(i):
    cor_ans = get_correct_answer(i)
    cur, cor = getOptionsString(get_row_options(i), True, cor_ans)
    
    query = ''
    query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + cur
    query += '\nAs an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is '
    return query, cor

def get_peturbed_row_query(i, K):
    query = ''

    query += 'Question -\n\n' + perturb_input(get_row_question(i), K) + '\n\nChoices -\n\n' + getOptionsString(get_row_options(i))
    query += '\nAs an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is '

    # print(query)
    return query

def getOptionsDict(row):
    options = get_row_options(row)
    if isinstance(options, str):
        options = strToDict(options)
    return options
    
def getOptionsArr(row):
    cur = getOptionsDict(row)
    return [(i, cur[i]) for i in cur]
    
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
def getOptionsString(options, randomized=False, cor_ans = None):
    if isinstance(options, str):
        options = strToDict(options)
    options = [(i, options[i]) for i in options]
        
    if randomized:
        random.shuffle(options)
        
    shuffled = []
    res = ''
    idx = 0
    for i in options:  
        shuffled.append(i[0])
        res += choices[idx] + ' ' + i[1].split('\n')[0]
        res += '\n'
        idx += 1

    if randomized:
        return (res, shuffled)
    return res

def get_ith_option(row, i):
    options = get_row_options(row)
    
    if isinstance(options, str):
        options = strToDict(options)
    
    options = [options[j] for j in options]

    return options[i].split('\n')[0]


# Function to convert a string to a dictionary
def strToDict(Str):
    try:
        return ast.literal_eval(Str)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dictionary: {e}")
        return {}

def perturb_input(input_text, edit_distance):
    """
    Randomly perturb the input text by the specified edit distance.
    """
    chars = list(input_text)
    for _ in range(edit_distance):
        operation = random.choice(['insert', 'delete', 'substitute'])
        if operation == 'insert':
            pos = random.randint(0, len(chars))
            chars.insert(pos, random.choice(string.ascii_letters))
        elif operation == 'delete' and chars:
            pos = random.randint(0, len(chars) - 1)
            chars.pop(pos)
        elif operation == 'substitute' and chars:
            pos = random.randint(0, len(chars) - 1)
            chars[pos] = random.choice(string.ascii_letters)
    return ''.join(chars)


def get_compare_query_func(row):
    cur = getOptionsDict(row)
    
    def get(i, j):
        cur_dict = {}
        
        cur_dict[choices[i]] = cur[choices[i]]
        cur_dict[choices[j]] = cur[choices[j]]

        query = ''
        query += 'Question -\n\n' + get_row_question(i) + '\n\nChoices -\n\n' + getOptionsString(cur_dict, False)
        # query += '\n\nAs an extremely experienced and knowledgeable medical professional, which response is more appropriate? Respond with A or B.'
        query += '\n\nnAs an extremely experienced and knowledgeable medical professional, which response is more appropriate? Respond with A or B. \n\n Solution - '
        
        # print(query)
        return query
        # return "McDonalds more healthy than apple. Answer True or False."
    return get

def shuffle():
    reset_random_state()

def reset_random_state():
    global random_state
    random_state = random.randint(0, 2**32 - 1)
    random.seed(random_state)

if __name__ == '__main__':
    # test = get_compare_query_func(0)
    # print(test(2, 1))
    # print(test(1, 2))
    # print(get_is_correct_query(97))
    # print(get_row_query(97))
    # print(get_correct_answer(97))
    
    # high_entropy_correct = [9904, 13, 2147,
    #                         1842, 1737, 3212, 1521, 754, 483, 3390]
    
    high_entropy_correct = [2497, 2577, 1080]
    
    for i in high_entropy_correct:
        print(get_row_query(i))
        print(get_correct_answer(i))
        print('\n')

    

    # low_entropy_incorrect = [11273, 3086, 12453, 9482, 14119, 14031, 9444, 2143, 3832, 160]
    
    # low_entropy_incorrect = [11273, 3086, 12453]
    
    # for i in low_entropy_incorrect:
    #     # print(getOptionsArr(i))
    #     print(get_row_query(i))
    #     print(get_correct_answer(i))
    #     print('\n')
    
    # print(get_row_query(0))
    
    
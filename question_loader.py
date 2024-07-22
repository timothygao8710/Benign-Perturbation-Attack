import pandas as pd
import ast

FEW_SHOT = """

######################
-Examples-
######################
Text:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed by a press conference where Fed Chair Jerome Powell will take questions. Investors expect the Federal Open Market Committee to hold its benchmark interest rate steady in a range of 5.25%-5.5%.
######################
Output:
("entity"{{tuple_delimiter}}FED{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Fed is the Federal Reserve, which is setting interest rates on Tuesday and Wednesday)
{{record_delimiter}}
("entity"{{tuple_delimiter}}JEROME POWELL{{tuple_delimiter}}PERSON{{tuple_delimiter}}Jerome Powell is the chair of the Federal Reserve)
{{record_delimiter}}
("entity"{{tuple_delimiter}}FEDERAL OPEN MARKET COMMITTEE{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}The Federal Reserve committee makes key decisions about interest rates and the growth of the United States money supply)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}JEROME POWELL{{tuple_delimiter}}FED{{tuple_delimiter}}Jerome Powell is the Chair of the Federal Reserve and will answer questions at a press conference{{tuple_delimiter}}9)
{{completion_delimiter}}
######################
Text:
Arm's (ARM) stock skyrocketed in its opening day on the Nasdaq Thursday. But IPO experts warn that the British chipmaker's debut on the public markets isn't indicative of how other newly listed companies may perform.

Arm, a formerly public company, was taken private by SoftBank in 2016. The well-established chip designer says it powers 99% of premium smartphones.
######################
Output:

######################
-Real Data-
######################
Text: {input_text}
######################

"""

# Load the data
data = pd.read_json(path_or_buf='./data_clean/questions/US/US_qbank.jsonl', lines=True)
print(data.head())

# Function to get the length of the data
def get_data_len():
    return len(data)

# Function to get the query of a specific row
def get_row_query(i):
    query = 'Question -\n\n' + get_row_question(i) + '\nChoices -\n' + getOptionsString(get_row_options(i))
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
        res += i + ' ' + options[i]
        res += '\n'
    return res

# Function to convert a string to a dictionary
def strToDict(Str):
    try:
        return ast.literal_eval(Str)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dictionary: {e}")
        return {}

if __name__ == '__main__':
    print(get_row_query(9))

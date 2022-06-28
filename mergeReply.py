import pandas as pd
import ast

def reply_to_paragraph(Reply):
    return " ".join(list(reversed(ast.literal_eval(Reply))))

def merge_reply(Data):
    Output = pd.DataFrame(columns=['Question', 'Reply'])

    for i in Data.index:
        Reply = reply_to_paragraph(Data.at[i, 'Reply'])
        Question = Data.at[i, 'Question']
        if(Reply == '' or Question == '' ):
            Question = pd.NA
        Output.loc[len(Output.index)] = [Question, Reply]

    #drop column with Nan
    Output = Output.dropna(how='any')

    return Output

def merge_df(data1, data2):
    data = pd.DataFrame(columns=data1.columns)
    idx = 0
    for i in data1.index:
        data.loc[idx] = [data1['Question'][i], data1['Reply'][i]]
        idx += 1

    for i in data2.index:
        data.loc[idx] = [data2['Question'][i], data2['Reply'][i]]
        idx += 1
    
    return data
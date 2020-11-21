from __future__ import print_function

import os, json
import pandas as pd
import re

def remove_num(input_Str):
    pattern = '[0-9]'
    return re.sub(pattern, '', input_Str)

if __name__ == "__main__":
    pdf_path = '/home/hchen28/document_parses/pdf_json/'
    pdf_files = [j_ for j_ in os.listdir(pdf_path) if j_.endswith('.json')]

    filename = 0
    for pdf_file in pdf_files:
        path = pdf_path + pdf_file
        with open(path) as f:
            data = json.load(f)
            pid = data['paper_id']
            bt = ''
            for txt in data['body_text']:
                bt = bt + remove_num(txt['text'])
                bt = bt + ' '
            d1 = {'paper_id': pid, 'body_text': bt}
            fn = '/Users/rafaelchen/Desktop/assignment3/converted/pdf/' + str(filename) + '.json'
            try:
                with open(fn, 'a') as output:
                    json.dump(d1, output)
            except IOError:
                with open(fn, 'w+') as output:
                    json.dump(d1, output)
            if os.stat(fn).st_size > 1024 * 1024:
                filename = filename + 1
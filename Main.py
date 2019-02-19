#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pandas as pd


def export_csv():
    messages = []
    tags = []

    script_dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(script_dir + '/Spam'):
        print('Reading files...')
        for f in files:
            with open('Spam/' + f, 'r', encoding='utf-8', errors='ignore') as data:
                print('Reading ' + str(f) + '....')
                content = data.read()
                file_messages = content.split('###################################################################')
                for m in file_messages:
                    tmp = list(m)
                    tmp[0] = tmp[len(tmp) - 1] = ''
                    tmp = ''.join(tmp)
                    messages.append(tmp)
                    tags.append('Spam')

    with open('NonSpam/non_spam.txt', 'r', encoding='utf-8', errors='ignore') as data:
        print('Reading Non Spam Messages...')
        content = data.read()
        file_lines = content.split('\n')
        index = 1
        for line in file_lines:
            if index % 3 == 2:
                messages.append(line)
                tags.append('Non Spam')
            index += 1

    mes = pd.Series(messages)
    tag = pd.Series(tags)
    data_frame = pd.DataFrame({'Message': mes, 'Tag': tag})
    data_frame.to_csv('Dataset.csv', encoding='utf-8')


def export_files():

    i = 1
    script_dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(script_dir + '/Spam'):
        print('Reading files...')
        for f in files:
            with open('Spam/' + f, 'r') as data:
                print('Reading ' + str(f) + '....')
                content = data.read()
                file_messages = content.split('###################################################################')
                for m in file_messages:
                    with open('Dataset/' + str(i) + '.txt', 'w') as out:
                        tmp = list(m)
                        tmp[0] = tmp[len(tmp) - 1] = ''
                        tmp = ''.join(tmp)
                        out.write(tmp)
                        print('Exporting file ' + str(i) + '.txt...')
                        i += 1


if __name__ == "__main__":
    export_csv()
    # export_files()

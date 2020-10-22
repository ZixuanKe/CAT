import numpy as np
import json
import os
import torch


size=[1,28,28]


x = {}
y = {}

for file in os.listdir('./'):
    if '.py' not in file:
        # print('file: ',file)
        with open(file) as json_file:
            data = json.load(json_file) # read file and do whatever we need to do.
            # print('users: ',data['users'])
            # print('num_samples: ',data['num_samples'])
            for key, value in data['user_data'].items():
                # print('user: ',key)
                # print('length: ',len(value))
                for type, data in value.items():
                    if type == 'x':
                        if key in x:
                            print('repeat')
                            x[key].append(torch.from_numpy(np.array(data)))
                        elif key not in x:
                            x[key] = [torch.from_numpy(np.array(data))]

                    elif type == 'y':
                        if key in y:
                            y[key].append(data)
                        elif key not in y:
                            y[key] = [data]


# You can choose to further group users with the dictinary set up here

# print('x key: ',list(x.keys()))
# print('x length: ',len(list(x.keys())))
# print('y key: ',list(y.keys()))
# print('y length: ',len(list(y.keys())))

# print('x: ',x.size())
# print('y: ',y.size())
# print('user: ',len(user))
#
#
# sample = {'user': user[0], 'x': x[0], 'y': y[0]}
# print('sample: ',sample)


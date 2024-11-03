

# persons_dirs = {'Avihoo':{'train': train_Avihoo_dir, 'test': test_Avihoo_dir}, 'Tom': {'train': train_Tom_dir, 'test': test_Tom_dir},
#                 'Ohad': {'train': train_Ohad_dir, 'test': test_Ohad_dir},
#                 'Shai': {'train': train_Shai_dir, 'test': test_Shai_dir},
#                 'Aviner':  {'train': train_Aviner_dir,'test':test_Aviner_dir},
#                 'Daniel': {'train': train_Daniel_dir,'test':test_Daniel_dir},
#                 'Foad':{'train':train_Foad_dir,'test':test_Foad_dir},
#                 'Pery':{'train':train_Pery_dir,'test':test_Pery_dir},
#                 'Ariel': {'train':train_Ariel_dir, 'test': test_Ariel_dir},
#                 'Ofek':{'train':train_Ofek_dir, 'test': test_Ofek_dir}}

# one_persons_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\17_7_2024_DB\One_person"
one_persons_dir = '/home/wld-algo-6/Data/One_person'
persons = ['Alex', 'Avihoo', 'Tom', 'Ohad','Shai','Anton',
                'Aviner','Hadas',
                'Daniel','Itai',#'Leeor',
                'Foad',
                'Perry',
                'Ariel',
                'Ofek']

import os
# os.path.join(one_persons_dir,'Alex','Train')
persons_dirs = {person:{'train': os.path.join(one_persons_dir,person,'Train'),
                        'test': os.path.join(one_persons_dir,person,'Test')} for person in persons}
ttt = 1
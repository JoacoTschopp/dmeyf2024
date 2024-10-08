import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Configura la ID de la competencia y la lista de archivos con sus descripciones
competition = 'dm-ey-f-2024-primera'
scores_dir = './scores'
experiment_name = 'KA5720SA'

files_dir = './exp/' + experiment_name

submission_description = 'DESCIPCION: Experimento 3 SA- Sin atributos entreno solo con 202104'

# Inicializar la API usando las credenciales de kaggle.json
api = KaggleApi()
api.authenticate()

files = os.listdir(files_dir)
files = [f for f in files if f.endswith('.csv')]

# ordenar files por el numero antes de .csv
files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[2]))

# Lista de archivos a subir
submissions = [{'file': f'{files_dir}/{f}',
                'description': f'{submission_description}'} for f in files]

# Subir los archivos a la competencia
for submission in submissions:
    file_path = submission['file']
    description = submission['description']
    if os.path.exists(file_path):
        print(f'Subiendo {file_path}...')

        error = True

        while error:
            try:
                api.competition_submit(file_name=file_path,
                                       message=description,
                                       competition=competition)
            except Exception as e:
                print(f'Error al subir {file_path}: {e}')
            else:
                error = False

        print(f'{file_path} subido con éxito.')
    else:
        print(f'El archivo {file_path} no existe.')


# Get your submission history for the competition
submissions = api.competition_submissions(competition, page_size=100)

# Create a list of dictionaries to store submission details
submission_list = []

for submission in submissions:
    submission_info = {
        'SubmissionId': submission.ref,
        'FileName': submission.fileName,
        'Date': submission.date,
        'Score': submission.publicScore,
        'Description': submission.description
    }
    submission_list.append(submission_info)

# Convert the list into a DataFrame for better readability and manipulation
df = pd.DataFrame(submission_list)

# Optionally, save the scores to a CSV file
df.to_csv('./scores/my_kaggle_submissions.csv', index=False)

print("Submission scores saved to 'my_kaggle_submissions.csv'")

df = pd.read_csv('./scores/my_kaggle_submissions.csv')
df.head()

# extract the number of sends from the file name
df['sends'] = df['FileName'].str.extract(r'_(\d+).csv').astype(int)



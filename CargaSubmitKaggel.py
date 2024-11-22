import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import time

os.environ['KAGGLE_CONFIG_DIR'] = '/home/joaquintschopp/buckets/b1'
# Configura la ID de la competencia y la lista de archivos con sus descripciones
competition = 'dm-ey-f-2024-segunda'
scores_dir = '/home/joaquintschopp/buckets/b1/scores'
experiment_name = 'Kaggle02_20.2'

files_dir = '/home/joaquintschopp/buckets/b1/exp/Cortes'

submission_description = 'DESCIPCION: Semillero 20.2'

# Inicializar la API usando las credenciales de kaggle.json
api = KaggleApi()
api.authenticate()

files = os.listdir(files_dir)
files = [f for f in files if f.endswith('.csv')]

# Ordenar los archivos por el número después de "sem_" y luego por el número después de "corte"
files = sorted(files, key=lambda x: (
    int(x.split('_')[3]),  # Número del semestre después de 'sem_'
    int(x.split('_')[5][5:])  # Número después de 'corte'
))

# Lista de archivos a subir
submissions = [{'file': os.path.join(files_dir, f),
                'description': submission_description} for f in files]

# Ordenar nuevamente por el mismo criterio en caso de necesitarlo
submissions = sorted(submissions, key=lambda x: (
    int(x['file'].split('/')[-1].split('_')[3]),  # Número del semestre después de 'sem_'
    int(x['file'].split('/')[-1].split('_')[5][5:])  # Número después de 'corte'
))


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

        # Pausa de 25 segundos entre cargas
        time.sleep(25)
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
df.to_csv(
    '/home/joaquintschopp/buckets/b1/scores/my_kaggle_submissions20.2.csv', index=False)

print("Submission scores saved to 'my_kaggle_submissions20.2.csv'")

df = pd.read_csv(
    '/home/joaquintschopp/buckets/b1/scores/my_kaggle_submissions20.2.csv')
df.head()

# extract the number of sends from the file name
df['sends'] = df['FileName'].str.extract(r'_(\d+).csv').astype(int)

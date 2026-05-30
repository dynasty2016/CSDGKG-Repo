import os
import csv

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INPUT_FILENAME = 'CARICOM_Case_Studies_for_SDGs.csv'
OUTPUT_FILENAME = 'CARICOM_Case_Studies_for_SDGs_embeddings.csv'

csvfile_in = open(INPUT_FILENAME, encoding="cp1252", newline='')
input_quad = csv.DictReader(csvfile_in)

csvfile_out = open(OUTPUT_FILENAME, "w", encoding="utf8", newline='')
fieldnames = ['code','description','title','description_embedding']
output_quad = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
output_quad.writeheader()

llm = OpenAI()

for row in input_quad:
    print(row['description'])

    description = row['description'].replace('\n', ' ')
    description_response = llm.embeddings.create(
        input=description,
        model="text-embedding-ada-002"
    )

    output_quad.writerow({
        'code': row['code'],
        'title': row['title'],
        'description': row['description'], 
        'description_embedding': description_response.data[0].embedding
    })

csvfile_in.close()
csvfile_out.close()
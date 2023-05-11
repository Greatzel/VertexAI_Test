from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform

@dsl.pipeline(name='public-bucket-pipeline')
def public_bucket_pipeline(project_id: str):
  
    # Define the training and validation dataset URIs
    training_data_uri = 'gs://public_bucket/training_data.csv'
    validation_data_uri = 'gs://public_bucket/validation_data.csv'

    # Define the container command to run
    container_command = f'python train.py --training_data_uri {training_data_uri} --validation_data_uri {validation_data_uri}'

    # Submit the custom job to Vertex AI
    aiplatform.CustomJob(
        display_name='public-bucket-training-job',
        container_uri='gcr.io/my-project/my-image',
        command=container_command,
        project=project_id
    ).run(sync=True)

if __name__ == '__main__':
    compiler.Compiler().compile(public_bucket_pipeline, 'public-bucket-pipeline.tar.gz')

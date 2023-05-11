from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform

@dsl.pipeline(name='public-bucket-pipeline')
def public_bucket_pipeline(project_id: str):
  
    # Define the training and validation dataset URIs
    training_data_uri = 'gs://public_bucket/training_data.csv'
    validation_data_uri = 'gs://public_bucket/validation_data.csv'

    # Define the container spec
    container_spec = aiplatform.gas.ContainerSpec(
        image_uri='gcr.io/devgm/my-image',
        command=['python', 'main_pipeline.py'],
        args=[
            '--training_data_uri', training_data_uri,
            '--validation_data_uri', validation_data_uri
        ]
    )

    # Define the training job
    training_job = aiplatform.TrainingJob(
        display_name='public-bucket-training-job',
        container_spec=container_spec,
        requirements=[
            'google-cloud-storage==1.38.0',
            'scikit-learn==0.24.2'
        ],
        model_serving_container_image_uri=None,
        model_display_name=None,
    )

    # Submit the training job to Vertex AI
    training_job.run(sync=True)

if __name__ == '__main__':
    compiler.Compiler().compile(public_bucket_pipeline, 'public-bucket-pipeline.tar.gz')

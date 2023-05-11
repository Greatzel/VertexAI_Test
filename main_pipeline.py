import os
import kfp
from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
)

@component
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    learning_rate: float = 0.01,
    num_epochs: int = 10,
):
    # Code to train the machine learning model using the provided training data
    # and hyperparameters. This can involve preprocessing the data, selecting
    # a model architecture, training the model, and evaluating its performance.
    # Once the model is trained, save it to the provided model output artifact.

@component
def deploy_model(
    model: Input[Model],
    serving_container_image_uri: str,
    endpoint: Output[Artifact],
):
    # Code to deploy the trained machine learning model to an endpoint for serving.
    # This can involve creating a new endpoint, specifying the container image to use,
    # and deploying the model to the endpoint. Once the model is deployed, save the
    # endpoint to the provided endpoint output artifact.

# Define the pipeline
@dsl.pipeline(
    name='basic-vertex-ai-pipeline',
    description='A simple Vertex AI pipeline that trains and deploys a machine learning model',
)
def pipeline(
    training_data_uri: str = 'gs://my-bucket/training_data',
    serving_container_image_uri: str = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest',
):
    # Define inputs and outputs for the pipeline
    training_data = Input(name='training_data', source=training_data_uri, dataset_type='bigquery')
    model = Output(name='model', artifact_type='model')
    endpoint = Output(name='endpoint', artifact_type='endpoint')

    # Define pipeline steps
    train = train_model(training_data=training_data, model=model)
    deploy = deploy_model(model=train.outputs['model'], serving_container_image_uri=serving_container_image_uri, endpoint=endpoint)

if __name__ == '__main__':
    # Compile and run the pipeline
    kfp.compiler.Compiler().compile(pipeline, 'pipeline.yaml')
    aiplatform.init(project='devgm')
    pipeline_job = aiplatform.PipelineJob(
        display_name='basic-vertex-ai-pipeline',
        template_path='pipeline.yaml',
        enable_caching=False,
    )
    pipeline_job.run(sync=True)

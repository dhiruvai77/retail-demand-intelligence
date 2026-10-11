# Retail Demand Intelligence - AWS Production Pipeline

## Purpose

Extend the existing local retail demand forecasting project into a practical AWS data and machine-learning pipeline.

## Existing project

The current project prepares retail demand data, creates forecasting features, trains LightGBM models, evaluates predictions, and serves forecasts through FastAPI.

## Target architecture

Retail data -> Amazon S3  
-> AWS Glue ETL and data validation  
-> Partitioned Parquet data  
-> Glue Data Catalog  
-> Athena SQL analytics  
-> SageMaker model training  
-> Model versioning  
-> SageMaker batch inference  
-> Forecast outputs in S3  
-> API or dashboard delivery

## Planned AWS services

- Amazon S3: data and model storage
- AWS Glue: ETL and data-quality validation
- Glue Data Catalog: table metadata
- Amazon Athena: SQL analytics
- Amazon SageMaker: model training and batch inference
- Step Functions: workflow orchestration
- CloudWatch: logs and monitoring
- Amazon ECR and ECS: future API deployment

## Initial design decisions

- Develop and test locally before using AWS resources.
- Use a batch-first architecture.
- Use one AWS region initially.
- Store curated data as partitioned Parquet.
- Partition primarily by store and date.
- Keep raw datasets outside Git.
- Do not use a continuously running SageMaker endpoint initially.
- Reuse the existing data preparation, feature engineering, and model logic where possible.

## Planned S3 layout

```text
s3://<bucket-name>/
  raw/
  curated/
  features/
  predictions/
  models/
  quality/
  monitoring/
  athena-results/
```
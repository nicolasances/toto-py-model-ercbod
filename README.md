# Model ecrbod
The `ecrbod` model determines the category of the expense based on its description

## Environment
This model runs as a containerized microservice. 

It needs the following environment variables: 
 * **TOTO_API_AUTH**: API authentication param to be used when authenticating to Toto APIs
 * **TOTO_HOST**: the host where APIs are reachable 
 * **TOTO_TMP_FOLDER**: a folder to store tmp files - This should be set in the Dockerfile
 * **TOTO_EVENTS_GCP_PROJECT_ID**: the Google Project Id for events in the current environment
 * **TOTO_ENV**: the Toto environment (dev, prod, ...)

## Predictions
This model generates predictions in a single way: 
 * **single**: will generate a prediction on demand for a single expense 

The model generates files when it trains and stores them in a temporary folder under:
```
> {TOTO_TMP_FOLDER}/ecrbod/<uuid>/
```
Under that folder, the model will save:
 * `history` - a file with the relevant downloaded history
 * `features` - a file with the built features for the model
 * `predictions` (only in the batch inference) - a file with the generated predictions
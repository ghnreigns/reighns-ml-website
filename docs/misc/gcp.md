## GCP Bucket 

This is a tutorial for myself on how to setup GCP Bucket and use python to upload/download files. These are references here[^stackoverflow].

## Create GCP Bucket

We first create GCP Bucket here by following the steps here[^GCP Bucket].

## Python Google Cloud Storage

Install the Cloud Client Libraries for Python for an individual API like Cloud Storage
```bash
!pip install --upgrade google-cloud-storage
```

## Install Cloud SDK

Install Cloud SDK which can be used to access Cloud Storage services from the command line and then do `gcloud auth application-default login`. Note that this command generates credentials for client libraries. 

The steps are detailed here[^install Cloud SDK].

After installation, we need to install `gcloud`.
```bash
!pip install gcloud
```

You can also install Cloud SDK in command line:

```bash
# Windows
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")

& $env:Temp\GoogleCloudSDKInstaller.exe
```

Follow the prompts to install the Cloud SDK. Note it will also ask you to set the default project. You can choose the project you want to use.

---

## Setup Service Account

We open `cmd` prompt, `cd` to the directory where you are working on, then type `gcloud auth login`.

Since I do not have a service account, we follow this link[^service account] and follow the steps using either the Cloud Consoler or Command Line. (I prefer the Cloud Console). The documentation is clear and you just need to follow the steps.

Create a service account key by the following:
1. In the Cloud Console, click the email address for the service account that you created.
2. Click Keys.
3. Click Add key, then click Create new key.
4. Click Create. A JSON key file is downloaded to your computer.
5. Click Close.

## Setup Autheticated Environment

Now you have a `json` file from previous step. Put the `json` file in a folder. Then everytime you start a terminal or new window, you can use 

```bash
$env:GOOGLE_APPLICATION_CREDENTIALS="$PATH$TO$JSON"
```

## Upload and Download Files

```python
from gcloud import storage

def return_bucket(project_id: str) -> List:
    """Return a list of buckets for a given project.

    Args:
        project_id (str): The project id.

    Returns:
        List: A list of buckets.
    """
    storage_client = storage.Client(project=project_id)
    buckets = list(storage_client.list_buckets())
    return buckets


def upload_to_bucket(
    source_file_name: str,
    destination_blob_name: str,
    bucket_name: str,
    project_id: str,
) -> str:
    """Uploads a file to the bucket and returns the public url.

    Args:
        source_file_name (str): The file in local that you want to upload.
        destination_blob_name (str): The name of the file in the bucket. To include full path.
        bucket_name (str): The name of the bucket.
    """

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        f"file {source_file_name} uploaded to bucket {bucket_name} successfully!"
    )
    return blob.public_url


def download_from_bucket(
    source_file_name: str,
    destination_blob_name: str,
    bucket_name: str,
    project_id: str,
) -> None:
    """Download file from GCP bucket.
    Just do the opposite of upload_to_bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.download_to_filename(source_file_name)

if __name__ == "__main__":
   PROJECT_ID = "Your Project ID"
   BUCKET_NAME = "Bucket Name"
   SOURCE_FILE_NAME = "Source File Name stored Locally"
   DESTINATION_BLOB_NAME = "Destination File Name in GCP Bucket"
   upload_to_bucket(
      SOURCE_FILE_NAME, DESTINATION_BLOB_NAME, BUCKET_NAME, PROJECT_ID
   )
   download_from_bucket(
      SOURCE_FILE_NAME, DESTINATION_BLOB_NAME, BUCKET_NAME, PROJECT_ID
   )
   ```

If you want to mass upload or download, you just need to create a loop as such:

```python
for file in os.listdir(path):
    upload_to_bucket(SOURCE_FILE_NAME, DESTINATION_BLOB_NAME, BUCKET_NAME, PROJECT_ID)
```

[^GCP Bucket]: https://cloud.google.com/storage/docs/creating-buckets
[^stackoverflow]: https://stackoverflow.com/questions/69959969/how-to-write-files-from-local-to-gcp-using-python
[^install Cloud SDK]: https://cloud.google.com/sdk/docs/quickstart
[^service account]: https://cloud.google.com/docs/authentication/getting-started#command-line
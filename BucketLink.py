import boto3

def get_s3_file_url(bucket_name, file_key):
    # Construct the URL of the file in the S3 bucket
    s3_client = boto3.client('s3')
    url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': file_key}, ExpiresIn=3600)
    return url

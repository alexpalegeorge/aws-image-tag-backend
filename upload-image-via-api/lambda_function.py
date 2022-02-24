import json
import base64
import boto3

def lambda_handler(event, context):
    
    # get binary data of image
    s3 = boto3.client('s3')
    get_content = event["content"]
    decode_content = base64.b64decode(get_content)
    
    # create file name based on current file count
    dynamodb = boto3.client('dynamodb')
    response = dynamodb.get_item(
        TableName='filename-counter',
        Key={
            'total': {'S': 'files'}
        }
    )
    file_count = response['Item']['count']['N']
    new_count = int(file_count) + 1
    hex_key = hex(new_count)
    name = str(hex_key) + '.jpg'
    
    # update count
    table = boto3.resource('dynamodb').Table('filename-counter')
    table.put_item(
        Item = {
            'total': 'files',
            'count': new_count
        }
    )
    
    # upload to s3 bucket
    s3_upload = s3.put_object(Bucket="fit5225-tagtag-upload-bucket", 
                              Key=name,
                              Body = decode_content)
    
    message = "successfully uploaded new image: " + name
    
    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }

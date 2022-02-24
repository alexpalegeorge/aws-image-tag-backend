import json
import boto3

BUCKET_NAME = "fit5225-tagtag-upload-bucket"

def lambda_handler(event, context):
    
    url = event['image_url']
    image_name = url.split('/')[-1]
    
    # delete image from bucket
    s3 = boto3.resource('s3')
    image_file = s3.Object(BUCKET_NAME, image_name).delete()
    
    # find all tags with url, delete all from dynamodb
    dynamodb = boto3.client('dynamodb')
    all_entries = dynamodb.query(
        TableName='tagtag-sorted',
        IndexName='image-index',
        KeyConditionExpression= 'image = :pk',
        ExpressionAttributeValues= {
            ':pk': {
                'S': image_name
            }
        },
        ProjectionExpression= 'image, tag'
    )['Items']
    
    if len(all_entries) > 0:
        for entry in all_entries: 
            delete_entry(entry)
    
    return {
        'statusCode': 200,
        'body': 'image ' + image_name + ' no longer exists'
    }

    
def delete_entry(entry):
    
    image = entry['image']['S']
    tag = entry['tag']['S']
    
    dynamodb = boto3.client('dynamodb')
    delete = dynamodb.delete_item(
        TableName='tagtag-sorted',
        Key={
            'tag': {
                'S': tag
            },
            'image': {
                'S': image
            }
        }
    )

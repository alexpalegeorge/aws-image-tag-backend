import json
import boto3

BASE_URL = "https://fit5225-tagtag-upload-bucket.s3.amazonaws.com/"

def lambda_handler(event, context):
    
    # sets the search tags with each allowed combination (tag1 is 
    # force-validated by api)
    search_tags = {}
    if event['tag2'] == "" and event['tag3'] == "":
        search_tags = {
            'S': event['tag1']
        }
    elif event['tag3'] == "":
        search_tags = {
            'S': event['tag1'],
            'S': event['tag2']
        }
    elif event['tag2'] == "" and not event['tag3'] == "":
        search_tags = {
            'S': event['tag1'],
            'S': event['tag3']
        }
    else:
        search_tags = {
            'S': event['tag1'],
            'S': event['tag2'],
            'S': event['tag3']
        }
    
    # get items from dynamodb
    dynamodb = boto3.client('dynamodb')
    response = dynamodb.query(
        TableName='tagtag-sorted',
        KeyConditionExpression= 'tag = :pk',
        ExpressionAttributeValues= {
            ':pk': search_tags
        },
        ProjectionExpression= 'image, tag'
    )
    
    matching_links = {'links': []}
    
    for i in response['Items']:
        image_name = i['image']['S']
        image_url = BASE_URL + image_name
        matching_links['links'].append(image_url)
    
    return {
        'statusCode': 200,
        'body': matching_links
    }   
